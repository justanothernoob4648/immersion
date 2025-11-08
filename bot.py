from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp
import asyncio
from dotenv import load_dotenv
from loguru import logger

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
    deliver_props: bool
    prop_channel: str
    supplemental_vocab: tuple[str, ...]
    stuck_timeout: float
    hesitation_tokens: int


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def _parse_vocab(raw: Optional[str]) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(word.strip() for word in raw.split(",") if word.strip())


def load_session_config() -> SessionConfig:
    return SessionConfig(
        scenario_prompt=os.getenv(
            "LANG_AGENT_SCENARIO", "Simulate a conversation with a waiter at a restaurant"
        ),
        native_language=os.getenv("LANG_AGENT_NATIVE", "English"),
        target_language=os.getenv("LANG_AGENT_TARGET", "Spanish"),
        learner_name=os.getenv("LANG_AGENT_LEARNER_NAME", "Learner"),
        proficiency=os.getenv("LANG_AGENT_PROFICIENCY", "beginner"),
        deliver_props=_env_bool("LANG_AGENT_DELIVER_PROPS", True),
        prop_channel=os.getenv("LANG_AGENT_PROP_CHANNEL", "markdown"),
        supplemental_vocab=_parse_vocab(os.getenv("LANG_AGENT_VOCAB")),
        stuck_timeout=_env_float("LANG_AGENT_STUCK_TIMEOUT", 10.0),
        hesitation_tokens=_env_int("LANG_AGENT_HESITATION_TOKENS", 3),
    )


def build_system_prompt(cfg: SessionConfig) -> str:
    props_clause = (
        "Surface immersive props such as menus, tickets, or signage in concise markdown, "
        "label them clearly, and keep them under 120 words."
        if cfg.deliver_props
        else "Focus on vivid verbal descriptions instead of standalone props."
    )

    vocab_clause = "".join(
        f"\n- Key vocabulary to recycle naturally: '{word}'" for word in cfg.supplemental_vocab
    )

    return (
        "You are an encouraging language coach conducting a role play with "
        f"{cfg.learner_name}. Stay in character for the user's scenario, answer primarily in "
        f"{cfg.target_language}, and keep your turns crisp (<12 seconds of speech).\n"
        f"{props_clause}\n"
        f"When the learner hesitates, explicitly mispronounces repeated syllables, or stays silent, "
        f"briefly restate the missing phrase in {cfg.native_language} and provide an immediate retry "
        f"prompt back in {cfg.target_language}. Avoid long grammar lectures mid-dialogue; instead, "
        "inject short meta tips (<15 words) between turns."
        f"{vocab_clause}\n"
        "Always close each turn with an explicit, actionable prompt that nudges the learner to respond verbally."
    )


def build_bootstrap_messages(cfg: SessionConfig) -> list[dict[str, str]]:
    system_prompt = build_system_prompt(cfg)
    scenario_brief = (
        "Scenario briefing:\n"
        f"- Learner proficiency: {cfg.proficiency}\n"
        f"- Requested scene: {cfg.scenario_prompt}\n"
        f"- Native language: {cfg.native_language}\n"
        f"- Target language: {cfg.target_language}\n"
        f"- Output props as: {cfg.prop_channel}\n"
        "Start by greeting the learner, summarizing the scene, and sharing the first prop/stimulus."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": scenario_brief},
    ]


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


async def run_bot(webrtc_connection: Any) -> None:
    """Entry point invoked by server.py when a WebRTC connection is ready."""

    cfg = load_session_config()
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            audio_out_10ms_chunks=_env_int("LANG_AGENT_AUDIO_CHUNKS", 2),
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=_env_float("LANG_AGENT_VAD_STOP_SECS", 0.4),
                    start_secs=_env_float("LANG_AGENT_VAD_START_SECS", 0.1),
                    energy_threshold=_env_float("LANG_AGENT_VAD_THRESHOLD", 0.5),
                )
            ),
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
            model=os.getenv("OPEN_ROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=_env_float("OPEN_ROUTER_TEMPERATURE", 0.6),
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
                coach,
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
