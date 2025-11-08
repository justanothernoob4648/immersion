from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tavus.video import TavusVideoService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc.transport import SmallWebRTCTransport

from .coaching import CoachSettings, ConversationCoach
from .config import AgentBuildConfig, LanguageLearningSession
from .prompts import build_bootstrap_messages


@dataclass(slots=True)
class PipelineBundle:
    """Exposes the assembled pipeline and major services for orchestration layers."""

    pipeline: Pipeline
    transport: SmallWebRTCTransport
    stt: DeepgramFluxSTTService
    tts: ElevenLabsTTSService
    llm: OpenAILLMService
    tavus: TavusVideoService
    transcript: TranscriptProcessor


class LanguageLearningAgent:
    """Factory that assembles the Pipecat pipeline for each conversation session."""

    def __init__(self, config: AgentBuildConfig) -> None:
        self._config = config

    def build(
        self,
        *,
        session: LanguageLearningSession,
        webrtc_connection: Any,
        tavus_session: Any,
    ) -> PipelineBundle:
        """Create the Pipecat pipeline for a single learner session."""

        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(energy_threshold=self._config.vad_energy_threshold),
                audio_out_10ms_chunks=self._config.audio_out_chunks,
            ),
        )

        stt = DeepgramFluxSTTService(api_key=self._config.deepgram_api_key)
        tts = ElevenLabsTTSService(
            api_key=self._config.elevenlabs_api_key,
            voice_id=self._config.elevenlabs_voice_id,
        )

        llm = OpenAILLMService(
            api_key=self._config.openrouter_api_key,
            base_url=self._config.openrouter_base_url,
            model=self._config.openrouter_model,
            temperature=self._config.llm_temperature,
            max_output_tokens=self._config.llm_max_output_tokens,
        )

        tavus = TavusVideoService(
            api_key=self._config.tavus_api_key,
            replica_id=self._config.tavus_replica_id,
            session=tavus_session,
        )

        context = LLMContext(build_bootstrap_messages(session))
        context_aggregator = LLMContextAggregatorPair(context)
        transcript = TranscriptProcessor()
        coach = ConversationCoach(
            session=session,
            settings=CoachSettings(
                stuck_timeout_seconds=self._config.stuck_timeout_seconds,
                hesitation_token_threshold=self._config.hesitant_utterance_tokens,
            ),
        )

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
                transcript.assistant_tts(),
                context_aggregator.assistant(),
            ]
        )

        return PipelineBundle(
            pipeline=pipeline,
            transport=transport,
            stt=stt,
            tts=tts,
            llm=llm,
            tavus=tavus,
            transcript=transcript,
        )


__all__ = [
    "LanguageLearningAgent",
    "PipelineBundle",
]
