from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp
import asyncio
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMRunFrame,
    LLMMessagesUpdateFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tavus.video import TavusVideoService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)


@dataclass(slots=True)
class SessionConfig:
    scenario_prompt: str
    native_language: str
    target_language: str
    learner_name: str
    proficiency: str
    stuck_timeout: float
    hesitation_tokens: int
    stimulus_html: Optional[str] = None


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {key}: {raw}") from exc


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int for {key}: {raw}") from exc



def load_session_config() -> SessionConfig:
    return SessionConfig(
        scenario_prompt=os.getenv(
            "LANG_AGENT_SCENARIO", "Simulate a conversation with a waiter at a restaurant"
        ),
        native_language=os.getenv("LANG_AGENT_NATIVE", "English"),
        target_language=os.getenv("LANG_AGENT_TARGET", "Spanish"),
        learner_name=os.getenv("LANG_AGENT_LEARNER_NAME", "Learner"),
        proficiency=os.getenv("LANG_AGENT_PROFICIENCY", "beginner"),
        stuck_timeout=_env_float("LANG_AGENT_STUCK_TIMEOUT", 10.0),
        hesitation_tokens=_env_int("LANG_AGENT_HESITATION_TOKENS", 3),
    )


def build_system_prompt(cfg: SessionConfig) -> str:
    return (
        "You are an encouraging language coach conducting a role play with "
        f"{cfg.learner_name}. Stay in character for the user's scenario, answer primarily in "
        f"{cfg.target_language}, and keep your turns crisp (<12 seconds of speech).\n"
        f"Let the learner talk, and respond in {cfg.target_language}. If the learner resorts back to {cfg.native_language}"
        f"then offer them the correct pronounciation in {cfg.target_language}. Make sure to offer the user hints if they are"
        f"severely struggling (can't say a word, pronounciation is super bad, grammar is super bad). Otherwise, speak."
    )


def build_bootstrap_messages(cfg: SessionConfig) -> list[dict[str, str]]:
    system_prompt = build_system_prompt(cfg)
    scenario_brief = (
        "Scenario briefing:\n"
        f"- Learner proficiency: {cfg.proficiency}\n"
        f"- Requested scene: {cfg.scenario_prompt}\n"
        f"- Native language: {cfg.native_language}\n"
        f"- Target language: {cfg.target_language}\n"
        "Start by greeting the learner and summarizing the scene."
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": scenario_brief},
    ]

    if cfg.stimulus_html:
        # Provide the cleaned stimulus as read-only context.
        messages.append(
            {
                "role": "system",
                "content": (
                    "Stimulus context (read-only). Use this content to guide the scene.\n\n"
                    + cfg.stimulus_html
                ),
            }
        )

    return messages


class ConversationCoach(FrameProcessor):
    """Injects coaching cues when the learner is silent or hesitates."""

    def __init__(self, cfg: SessionConfig) -> None:
        super().__init__(name="conversation-coach")
        self._cfg = cfg
        self._last_user_activity = time.monotonic()
        self._watchdog_task: Optional[asyncio.Task[None]] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            return

        if not self._watchdog_task:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

        if isinstance(frame, TranscriptionFrame):
            speaker = getattr(frame, "speaker", getattr(frame, "participant", "user"))
            if speaker in {"user", "remote", None} and getattr(frame, "final", False):
                text = getattr(frame, "text", "").strip()
                self._last_user_activity = time.monotonic()
                if self._needs_hint(text):
                    await self._emit_hint(reason="hesitation", stuck_text=text)
        elif isinstance(frame, EndFrame):
            if self._watchdog_task:
                self._watchdog_task.cancel()
                self._watchdog_task = None
        await self.push_frame(frame, direction)

    async def _watchdog_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(max(self._cfg.stuck_timeout / 2, 1))
                if time.monotonic() - self._last_user_activity > self._cfg.stuck_timeout:
                    await self._emit_hint(reason="silence", stuck_text="")
        except asyncio.CancelledError:
            return

    def _needs_hint(self, text: str) -> bool:
        if not text:
            return False
        tokens = [token for token in text.replace("?", " ").split() if token]
        return len(tokens) <= self._cfg.hesitation_tokens

    async def _emit_hint(self, reason: str, stuck_text: str) -> None:
        cue = (
            f"Coach cue ({reason}): The learner may be stuck on '{stuck_text or 'unknown'}'."
            f" Offer a brief boost in {self._cfg.native_language}, then resume in {self._cfg.target_language}."
        )
        frame = LLMMessagesUpdateFrame(
            messages=[{"role": "system", "content": cue}],
            run_llm=True,
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)


async def run_bot(webrtc_connection: Any, *, stimulus_html: Optional[str] = None) -> None:
    """Entry point invoked by server.py when a WebRTC connection is ready."""

    cfg = load_session_config()
    # Attach stimulus, if provided via session
    cfg.stimulus_html = stimulus_html
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            audio_out_10ms_chunks=_env_int("LANG_AGENT_AUDIO_CHUNKS", 2),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
        ),
    )

    async with aiohttp.ClientSession() as http_session:
        stt = DeepgramFluxSTTService(api_key=_require_env("DEEPGRAM_API_KEY"))
        tts = ElevenLabsTTSService(
            api_key=_require_env("ELEVENLABS_API_KEY"),
            voice_id=_require_env("ELEVENLABS_VOICE_ID"),
        )
        llm = OpenAILLMService(
            api_key=_require_env("OPEN_ROUTER_API_KEY"),
            base_url=os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPEN_ROUTER_MODEL", "google/gemini-2.5-flash"),
            temperature=_env_float("OPEN_ROUTER_TEMPERATURE", 1.0),
            max_output_tokens=_env_int("OPEN_ROUTER_MAX_TOKENS", 400),
        )
        tavus = TavusVideoService(
            api_key=_require_env("TAVUS_API_KEY"),
            replica_id=_require_env("TAVUS_REPLICA_ID"),
            session=http_session,
        )

        context = LLMContext(build_bootstrap_messages(cfg))
        context_aggregator = LLMContextAggregatorPair(context)
        transcript = TranscriptProcessor()
        coach = ConversationCoach(cfg)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                transcript.user(),
                context_aggregator.user(),
                llm,
                tts,
                tavus,
                transport.output(),
                transcript.assistant(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_client_connected")
        async def _on_client_connected(tr, client):  # noqa: ANN001
            logger.info("Client connected")
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(tr, client):  # noqa: ANN001
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
