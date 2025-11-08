from __future__ import annotations

import asyncio
import os
from dataclasses import fields, replace
from typing import Any, Mapping

import aiohttp
from loguru import logger

from .config import LanguageLearningSession
from .runtime import build_agent_from_env


def _env_bool(var: str, default: bool) -> bool:
    raw = os.getenv(var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_vocab(var: str) -> tuple[str, ...]:
    if not var:
        return ()
    return tuple(word.strip() for word in var.split(",") if word.strip())


def session_from_env() -> LanguageLearningSession:
    """Build a LanguageLearningSession using optional environment overrides."""

    return LanguageLearningSession(
        scenario_prompt=os.getenv(
            "LANG_AGENT_SCENARIO", "Simulate a conversation with a waiter at a restaurant"
        ),
        native_language=os.getenv("LANG_AGENT_NATIVE", "English"),
        target_language=os.getenv("LANG_AGENT_TARGET", "Spanish"),
        learner_name=os.getenv("LANG_AGENT_LEARNER_NAME", "Learner"),
        proficiency=os.getenv("LANG_AGENT_PROFICIENCY", "beginner"),
        deliver_props=_env_bool("LANG_AGENT_DELIVER_PROPS", True),
        prop_channel=os.getenv("LANG_AGENT_PROP_CHANNEL", "markdown"),
        supplemental_vocab=_parse_vocab(os.getenv("LANG_AGENT_VOCAB", "")),
    )


async def create_tavus_session() -> Mapping[str, Any]:
    """Create a Tavus persona session via Tavus' REST API."""

    api_key = os.getenv("TAVUS_API_KEY")
    replica_id = os.getenv("TAVUS_REPLICA_ID")
    if not api_key or not replica_id:
        raise RuntimeError("TAVUS_API_KEY and TAVUS_REPLICA_ID must be set")

    base_url = os.getenv("TAVUS_API_BASE_URL", "https://api.tavus.io")
    session_endpoint = f"{base_url.rstrip('/')}/v2/replicas/{replica_id}/sessions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(headers=headers) as http:
        async with http.post(session_endpoint, json={}) as response:
            response.raise_for_status()
            payload = await response.json()
            logger.debug("Created Tavus session: {}", payload.get("id", payload))
            return payload


def _merge_session(
    base: LanguageLearningSession, overrides: Mapping[str, Any] | None
) -> LanguageLearningSession:
    if not overrides:
        return base
    allowed = {field.name for field in fields(LanguageLearningSession)}
    filtered = {key: value for key, value in overrides.items() if key in allowed}
    return replace(base, **filtered)


async def run_bot(
    webrtc_connection: Any,
    *,
    session_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Instantiate the Pipecat pipeline for a SmallWebRTC connection."""

    session = _merge_session(session_from_env(), session_overrides)
    tavus_session = await create_tavus_session()

    bundle = build_agent_from_env(
        scenario_prompt=session.scenario_prompt,
        native_language=session.native_language,
        target_language=session.target_language,
        learner_name=session.learner_name,
        proficiency=session.proficiency,
        supplemental_vocab=session.supplemental_vocab,
        deliver_props=session.deliver_props,
        prop_channel=session.prop_channel,
        webrtc_connection=webrtc_connection,
        tavus_session=tavus_session,
    )

    logger.info(
        "Starting Pipecat pipeline for learner '{}', scenario '{}'",
        session.learner_name,
        session.scenario_prompt,
    )

    try:
        await bundle.pipeline.run()
    except Exception:
        logger.exception("Pipeline crashed")
        raise
    finally:
        logger.info("Pipeline finished for learner '{}'", session.learner_name)


async def main() -> None:
    from pipecat.transports.network.small_webrtc.connection import SmallWebRTCConnection

    host = os.getenv("SMALLWEB_HOST")
    token = os.getenv("SMALLWEB_TOKEN")
    room_id = os.getenv("SMALLWEB_ROOM")
    if not all([host, token, room_id]):
        raise RuntimeError("SMALLWEB_HOST, SMALLWEB_TOKEN, and SMALLWEB_ROOM must be set")

    connection = SmallWebRTCConnection(
        host=host,
        token=token,
        room_id=room_id,
    )
    await run_bot(connection)


if __name__ == "__main__":
    asyncio.run(main())
