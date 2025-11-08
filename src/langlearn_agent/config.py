from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(slots=True)
class AgentBuildConfig:
    """Static configuration needed to assemble the Pipecat pipeline."""

    deepgram_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    openrouter_api_key: str
    tavus_api_key: str
    tavus_replica_id: str
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_temperature: float = 0.6
    llm_max_output_tokens: int = 400
    audio_out_chunks: int = 2
    vad_energy_threshold: float = 0.5
    stuck_timeout_seconds: float = 10.0
    hesitant_utterance_tokens: int = 3
    enable_debug_logging: bool = False

    @classmethod
    def from_env(cls, *, dotenv_path: str | None = None) -> "AgentBuildConfig":
        """Load credentials + tuning knobs from environment variables (supports .env)."""

        load_dotenv(dotenv_path)

        def read_env(key: str) -> str:
            value = os.getenv(key)
            if not value:
                raise RuntimeError(f"Missing required environment variable: {key}")
            return value

        return cls(
            deepgram_api_key=read_env("DEEPGRAM_API_KEY"),
            elevenlabs_api_key=read_env("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=read_env("ELEVENLABS_VOICE_ID"),
            openrouter_api_key=read_env("OPEN_ROUTER_API_KEY"),
            tavus_api_key=read_env("TAVUS_API_KEY"),
            tavus_replica_id=read_env("TAVUS_REPLICA_ID"),
        )


@dataclass(slots=True)
class LanguageLearningSession:
    """Run-time session values provided by the user or orchestration layer."""

    scenario_prompt: str
    native_language: str
    target_language: str
    learner_name: str = "Learner"
    proficiency: str = "beginner"
    deliver_props: bool = True
    prop_channel: str = "markdown"
    supplemental_vocab: tuple[str, ...] = field(default_factory=tuple)
    allow_native_language_replay: bool = True


__all__ = [
    "AgentBuildConfig",
    "LanguageLearningSession",
]
