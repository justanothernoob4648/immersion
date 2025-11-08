from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from pipecat.frames.frames import EndFrame, Frame, MessageFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .config import LanguageLearningSession


@dataclass(slots=True)
class CoachSettings:
    """Lightweight tuning knobs for the coaching heuristics."""

    stuck_timeout_seconds: float
    hesitation_token_threshold: int


class ConversationCoach(FrameProcessor):
    """Injects coaching cues when the learner is silent or hesitates."""

    def __init__(
        self,
        session: LanguageLearningSession,
        settings: CoachSettings,
    ) -> None:
        super().__init__(name="conversation-coach")
        self._session = session
        self._settings = settings
        self._last_user_activity = time.monotonic()
        self._watchdog_task: Optional[asyncio.Task[None]] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if not self._watchdog_task:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

        if isinstance(frame, TranscriptionFrame):
            speaker = getattr(frame, "speaker", getattr(frame, "participant", "user"))
            if speaker in {"user", "remote", None}:
                if getattr(frame, "final", False):
                    text = getattr(frame, "text", "").strip()
                    self._last_user_activity = time.monotonic()
                    if self._needs_hint(text):
                        await self._emit_hint(
                            reason="hesitation",
                            stuck_text=text,
                        )
        elif isinstance(frame, EndFrame):
            if self._watchdog_task:
                self._watchdog_task.cancel()
                self._watchdog_task = None
        await self.push_frame(frame, direction)

    async def _watchdog_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(max(self._settings.stuck_timeout_seconds / 2, 1))
                if time.monotonic() - self._last_user_activity > self._settings.stuck_timeout_seconds:
                    await self._emit_hint(reason="silence", stuck_text="")
        except asyncio.CancelledError:
            return

    def _needs_hint(self, text: str) -> bool:
        if not text:
            return False
        tokens = [token for token in text.replace("?", " ").split() if token]
        return len(tokens) <= self._settings.hesitation_token_threshold

    async def _emit_hint(self, reason: str, stuck_text: str) -> None:
        cue = (
            f"Coach cue ({reason}): The learner may be stuck on '{stuck_text or 'unknown'}'. "
            f"Offer a brief boost in {self._session.native_language}, then resume in "
            f"{self._session.target_language}."
        )
        message = MessageFrame(
            messages=[
                {
                    "role": "system",
                    "content": cue,
                }
            ]
        )
        await self.push_frame(message, FrameDirection.DOWNSTREAM)


__all__ = [
    "CoachSettings",
    "ConversationCoach",
]
