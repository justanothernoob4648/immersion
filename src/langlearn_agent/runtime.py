from __future__ import annotations

from typing import Any, Iterable

from .config import AgentBuildConfig, LanguageLearningSession
from .pipeline import LanguageLearningAgent, PipelineBundle


def build_agent_from_env(
    *,
    scenario_prompt: str,
    native_language: str,
    target_language: str,
    learner_name: str = "Learner",
    proficiency: str = "beginner",
    supplemental_vocab: Iterable[str] | None = None,
    deliver_props: bool = True,
    prop_channel: str = "markdown",
    dotenv_path: str | None = None,
    webrtc_connection: Any,
    tavus_session: Any,
) -> PipelineBundle:
    """Helper that loads config from env vars and returns a ready pipeline."""

    config = AgentBuildConfig.from_env(dotenv_path=dotenv_path)

    session = LanguageLearningSession(
        scenario_prompt=scenario_prompt,
        native_language=native_language,
        target_language=target_language,
        learner_name=learner_name,
        proficiency=proficiency,
        supplemental_vocab=tuple(supplemental_vocab or ()),
        deliver_props=deliver_props,
        prop_channel=prop_channel,
    )

    agent = LanguageLearningAgent(config)
    return agent.build(
        session=session,
        webrtc_connection=webrtc_connection,
        tavus_session=tavus_session,
    )


__all__ = [
    "build_agent_from_env",
]
