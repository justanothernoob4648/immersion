from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import aiohttp
from aiohttp import ClientTimeout
from dotenv import load_dotenv
from langdetect import DetectorFactory, LangDetectException, detect
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
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

DetectorFactory.seed = 0
sentiment_analyzer = SentimentIntensityAnalyzer()

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
    milestones: tuple["Milestone", ...]


@dataclass(frozen=True)
class Milestone:
    label: str
    keywords: tuple[str, ...]


@dataclass
class AnalyticsRecorder:
    conversation_log: list[dict[str, Optional[str]]]
    stt_records: list[dict[str, Any]]


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


def _parse_milestones(raw: Optional[str]) -> tuple[Milestone, ...]:
    if not raw:
        return ()
    milestones: list[Milestone] = []
    for chunk in raw.split(";"):
        entry = chunk.strip()
        if not entry:
            continue
        if ":" in entry:
            label, keywords_part = entry.split(":", 1)
        else:
            label, keywords_part = entry, entry
        keywords = tuple(
            kw.strip().lower() for kw in keywords_part.split(",") if kw.strip()
        ) or (label.strip().lower(),)
        milestones.append(Milestone(label=label.strip(), keywords=keywords))
    return tuple(milestones)


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
        milestones=_parse_milestones(os.getenv("LANG_AGENT_MILESTONES")),
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
        self._hint_counts: dict[str, int] = {"hesitation": 0, "silence": 0}
        self._interruption_count = 0

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
        elif isinstance(frame, InterruptionFrame):
            self._interruption_count += 1

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
        self._hint_counts[reason] = self._hint_counts.get(reason, 0) + 1
        cue = (
            f"Coach cue ({reason}): The learner may be stuck on '{stuck_text or 'unknown'}'."
            f" Offer a brief boost in {self._cfg.native_language}, then resume in {self._cfg.target_language}."
        )
        frame = LLMMessagesUpdateFrame(
            messages=[{"role": "system", "content": cue}],
            run_llm=True,
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def analytics_snapshot(self) -> dict[str, Any]:
        total_hints = sum(self._hint_counts.values())
        return {
            "total_hints": total_hints,
            "hints_by_type": dict(self._hint_counts),
            "interruption_count": self._interruption_count,
        }


class STTAnalyticsTap(FrameProcessor):
    """Captures raw STT results for post-run analytics."""

    def __init__(self, recorder: AnalyticsRecorder) -> None:
        super().__init__(name="stt-analytics-tap")
        self._recorder = recorder

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self._recorder.stt_records.append(
                {
                    "text": frame.text,
                    "timestamp": frame.timestamp,
                    "result": frame.result,
                }
            )
        await self.push_frame(frame, direction)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


LANG_NAME_TO_ISO = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "brazilian portuguese": "pt",
    "chinese": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "korean": "ko",
    "hindi": "hi",
    "arabic": "ar",
    "russian": "ru",
    "turkish": "tr",
    "dutch": "nl",
}


def language_name_to_iso(name: str) -> str:
    key = name.strip().lower()
    if not key:
        return "und"
    return LANG_NAME_TO_ISO.get(key, key[:2])


def _detect_language_iso(text: str) -> Optional[str]:
    cleaned = text.strip()
    if len(cleaned) < 3:
        return None
    try:
        return detect(cleaned)
    except LangDetectException:
        return None


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_latency_metrics(log: list[dict[str, Optional[str]]]) -> dict[str, Any]:
    latencies: list[float] = []
    last_assistant: Optional[datetime] = None
    for entry in log:
        ts = _parse_timestamp(entry.get("timestamp"))
        if ts is None:
            continue
        if entry["role"] == "assistant":
            last_assistant = ts
        elif entry["role"] == "user" and last_assistant:
            delta = (ts - last_assistant).total_seconds()
            if delta >= 0:
                latencies.append(delta)
            last_assistant = None
    if not latencies:
        return {"samples": 0, "avg_seconds": 0.0, "max_seconds": 0.0, "long_pauses": 0}
    long_pauses = sum(1 for val in latencies if val >= 5.0)
    return {
        "samples": len(latencies),
        "avg_seconds": round(sum(latencies) / len(latencies), 2),
        "max_seconds": round(max(latencies), 2),
        "long_pauses": long_pauses,
    }


def compute_sentiment_metrics(log: list[dict[str, Optional[str]]]) -> dict[str, Any]:
    scores = []
    for entry in log:
        if entry["role"] != "user":
            continue
        text = (entry.get("content") or "").strip()
        if not text:
            continue
        comp = sentiment_analyzer.polarity_scores(text)["compound"]
        scores.append(comp)
    if not scores:
        return {"samples": 0, "avg_compound": 0.0, "trend": "neutral"}
    avg = sum(scores) / len(scores)
    trend = (
        "positive"
        if avg > 0.2
        else "negative"
        if avg < -0.2
        else "neutral"
    )
    return {"samples": len(scores), "avg_compound": round(avg, 3), "trend": trend}


def detect_error_clusters(log: list[dict[str, Optional[str]]]) -> dict[str, Any]:
    filler_words = {"um", "uh", "erm", "er", "eh", "like"}
    filler_count = 0
    short_responses = 0
    repeated_stems: Counter[str] = Counter()
    prev_text = None
    for entry in log:
        if entry["role"] != "user":
            continue
        text = (entry.get("content") or "").strip()
        tokens = _tokenize(text)
        if len(tokens) <= 2:
            short_responses += 1
        filler_count += sum(1 for token in tokens if token in filler_words)
        if prev_text and prev_text.lower() == text.lower():
            repeated_stems[text.lower()] += 1
        prev_text = text
    top_repeats = repeated_stems.most_common(3)
    return {
        "filler_words": filler_count,
        "short_responses": short_responses,
        "repeated_phrases": top_repeats,
    }


def _extract_word_confidences(result: Any) -> list[tuple[str, float]]:
    if not isinstance(result, dict):
        return []
    words = []
    alternatives = []
    if "channel" in result:
        alternatives = result["channel"].get("alternatives", [])
    elif "alternatives" in result:
        alternatives = result.get("alternatives", [])
    if alternatives:
        words = alternatives[0].get("words", [])
    confidences = []
    for word_obj in words:
        word_text = word_obj.get("word")
        conf = word_obj.get("confidence")
        if word_text and isinstance(conf, (int, float)):
            confidences.append((word_text, conf))
    return confidences


def analyze_pronunciation(records: list[dict[str, Any]]) -> dict[str, Any]:
    confidences: list[float] = []
    low_conf_words: list[tuple[str, float]] = []
    for rec in records:
        for word, conf in _extract_word_confidences(rec.get("result")):
            confidences.append(conf)
            if conf < 0.85:
                low_conf_words.append((word, conf))
    if not confidences:
        return {"average_confidence": None, "low_confidence_words": [], "low_confidence_count": 0}
    low_conf_words.sort(key=lambda item: item[1])
    return {
        "average_confidence": round(sum(confidences) / len(confidences), 3),
        "low_confidence_words": low_conf_words[:5],
        "low_confidence_count": len(low_conf_words),
    }


def evaluate_milestones(cfg: SessionConfig, log: list[dict[str, Optional[str]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not cfg.milestones:
        return results
    for milestone in cfg.milestones:
        hit_ts = None
        for entry in log:
            if entry["role"] != "user":
                continue
            content = (entry.get("content") or "").lower()
            if all(keyword in content for keyword in milestone.keywords):
                hit_ts = entry.get("timestamp")
                break
        results.append(
            {
                "label": milestone.label,
                "keywords": milestone.keywords,
                "achieved": hit_ts is not None,
                "timestamp": hit_ts,
            }
        )
    return results


def _message_stats(messages) -> dict[str, Any]:
    if not messages:
        return {
            "turns": 0,
            "total_words": 0,
            "avg_words_per_turn": 0.0,
            "unique_words": 0,
            "top_words": [],
            "recent_messages": [],
            "token_counter": Counter(),
        }

    word_counts: list[int] = []
    token_counter: Counter[str] = Counter()
    language_counter: Counter[str] = Counter()
    for msg in messages:
        text = (msg.content or "").strip()
        tokens = _tokenize(text)
        token_counter.update(tokens)
        word_counts.append(len(tokens))
        lang = _detect_language_iso(text)
        if lang:
            language_counter[lang] += len(tokens)

    total_words = sum(word_counts)
    avg_words = total_words / len(messages) if messages else 0.0
    recent = [
        {
            "timestamp": getattr(msg, "timestamp", None),
            "content": (msg.content or "").strip(),
        }
        for msg in messages[-3:]
    ]

    return {
        "turns": len(messages),
        "total_words": total_words,
        "avg_words_per_turn": round(avg_words, 2),
        "unique_words": len(token_counter),
        "top_words": token_counter.most_common(5),
        "recent_messages": recent,
        "token_counter": token_counter,
        "language_token_counts": language_counter,
    }


def generate_conversation_analytics(
    cfg: SessionConfig,
    transcript: TranscriptProcessor,
    coach: ConversationCoach,
    conversation_log: list[dict[str, Optional[str]]],
    stt_records: list[dict[str, Any]],
) -> dict[str, Any]:
    user_proc = transcript.user()
    assistant_proc = transcript.assistant()
    user_messages = list(getattr(user_proc, "_processed_messages", []))
    assistant_messages = list(getattr(assistant_proc, "_processed_messages", []))

    user_stats = _message_stats(user_messages)
    assistant_stats = _message_stats(assistant_messages)

    total_words = user_stats["total_words"] + assistant_stats["total_words"]
    talk_ratio = (
        user_stats["total_words"] / total_words if total_words else 0.0
    )

    user_tokens = user_stats["token_counter"]
    user_corpus = " ".join((msg.content or "").lower() for msg in user_messages)
    vocab_hits = [word for word in cfg.supplemental_vocab if word.lower() in user_corpus]
    vocab_misses = [word for word in cfg.supplemental_vocab if word.lower() not in user_corpus]

    user_language_counts = user_stats["language_token_counts"]
    assistant_language_counts = assistant_stats["language_token_counts"]
    combined_language_counts = user_language_counts + assistant_language_counts
    native_iso = language_name_to_iso(cfg.native_language)
    target_iso = language_name_to_iso(cfg.target_language)
    total_language_words = sum(combined_language_counts.values())
    native_words = combined_language_counts.get(native_iso, 0)
    target_words = combined_language_counts.get(target_iso, 0)
    other_words = total_language_words - native_words - target_words

    latency_stats = compute_latency_metrics(conversation_log)
    sentiment_stats = compute_sentiment_metrics(conversation_log)
    error_clusters = detect_error_clusters(conversation_log)
    pronunciation = analyze_pronunciation(stt_records)
    milestones = evaluate_milestones(cfg, conversation_log)
    user_total_words = user_stats["total_words"]
    vocab_diversity = (
        round(user_stats["unique_words"] / user_total_words, 3)
        if user_total_words
        else 0.0
    )

    analytics = {
        "conversation": {
            "total_turns": user_stats["turns"] + assistant_stats["turns"],
            "user_talk_ratio": round(talk_ratio, 2),
            "vocab_diversity": vocab_diversity,
        },
        "user": {k: v for k, v in user_stats.items() if k != "token_counter"},
        "assistant": {k: v for k, v in assistant_stats.items() if k != "token_counter"},
        "supplemental_vocab": {
            "covered": vocab_hits,
            "missed": vocab_misses,
        },
        "coach_hints": coach.analytics_snapshot(),
        "language_usage": {
            "native": {
                "language": cfg.native_language,
                "iso": native_iso,
                "approx_words": native_words,
                "share": round(native_words / total_language_words, 2) if total_language_words else 0.0,
            },
            "target": {
                "language": cfg.target_language,
                "iso": target_iso,
                "approx_words": target_words,
                "share": round(target_words / total_language_words, 2) if total_language_words else 0.0,
            },
            "other_words": other_words,
        },
        "latency": latency_stats,
        "sentiment": sentiment_stats,
        "error_clusters": error_clusters,
        "milestones": milestones,
        "pronunciation": pronunciation,
    }

    # Clean up helper counters before returning
    analytics["user"].pop("token_counter", None)
    analytics["assistant"].pop("token_counter", None)
    analytics["user"].pop("language_token_counts", None)
    analytics["assistant"].pop("language_token_counts", None)
    analytics["conversation_log_size"] = len(conversation_log)
    assistant_turns = max(assistant_stats["turns"], 1)
    coach_snapshot = analytics["coach_hints"]
    interruptions = coach_snapshot.get("interruption_count", 0)
    coach_snapshot["interruption_rate"] = round(interruptions / assistant_turns, 2)
    return analytics


async def request_post_session_feedback(
    http_session: aiohttp.ClientSession,
    cfg: SessionConfig,
    conversation_log: list[dict[str, Optional[str]]],
    analytics: dict[str, Any],
) -> None:
    if not conversation_log:
        return

    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        logger.warning("OPEN_ROUTER_API_KEY missing; skipping feedback generation")
        return

    model = os.getenv("OPEN_ROUTER_FEEDBACK_MODEL") or os.getenv(
        "OPEN_ROUTER_MODEL", "openai/gpt-4o-mini"
    )
    base_url = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

    formatted_transcript = "\n".join(
        f"[{entry.get('timestamp') or 'unknown'}] {entry['role']}: {entry.get('content', '')}"
        for entry in conversation_log[-40:]
    )

    prompt = (
        "You are a supportive language tutor reviewing a role-play between a learner"
        f" ({cfg.native_language} speaker) practicing {cfg.target_language}."
        " Provide concise written feedback covering: strengths, specific improvement"
        " areas (pronunciation, vocabulary, grammar), and one actionable exercise."
        " Keep it under 180 words."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Conversation transcript (most recent turns first):\n"
                    f"{formatted_transcript}\n\n"
                    "Session analytics: \n"
                    f"{json.dumps(analytics, ensure_ascii=False, indent=2)}\n"
                    "Write the feedback now."
                ),
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPEN_ROUTER_HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.getenv("OPEN_ROUTER_TITLE")
    if title:
        headers["X-Title"] = title

    try:
        response = await http_session.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=ClientTimeout(total=60),
        )
        response.raise_for_status()
        body = await response.json()
        feedback = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if feedback:
            logger.info("Post-session feedback:\n{}", feedback.strip())
    except Exception as exc:  # pragma: no cover - network operations
        logger.warning("Failed to fetch post-session feedback: {}", exc)


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

    recorder = AnalyticsRecorder(conversation_log=[], stt_records=[])

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
        @transcript.event_handler("on_transcript_update")
        async def _capture_transcript(_processor, frame):  # noqa: ANN001
            for message in frame.messages:
                recorder.conversation_log.append(
                    {
                        "role": message.role,
                        "content": (message.content or "").strip(),
                        "timestamp": message.timestamp,
                    }
                )
        coach = ConversationCoach(cfg)
        stt_tap = STTAnalyticsTap(recorder)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                stt_tap,
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
        try:
            await runner.run(task)
        finally:
            analytics = generate_conversation_analytics(
                cfg,
                transcript,
                coach,
                recorder.conversation_log,
                recorder.stt_records,
            )
            if analytics:
                logger.info(
                    "Conversation analytics:\n{}",
                    json.dumps(analytics, indent=2, ensure_ascii=False),
                )
                await request_post_session_feedback(
                    http_session=http_session,
                    cfg=cfg,
                    conversation_log=recorder.conversation_log,
                    analytics=analytics,
                )
