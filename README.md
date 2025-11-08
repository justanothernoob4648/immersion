# DEMO VIDEO

https://www.youtube.com/watch?v=kn-u5i2jlFM

# Language-Learning Agent (Pipecat)

Service-side implementation of a multimodal, role-play language tutor built on [Pipecat](https://pipecat.ai/). The agent accepts a learner prompt (scenario), their native language, and the target language, then drives the conversation with:

- **SmallWebRTCTransport** for bi-directional audio using the SmallWeb RTC transport.
- **DeepgramFluxSTTService** for streaming STT.
- **OpenRouter-backed OpenAILLMService** for the conversation brain.
- **ElevenLabsTTSService** for lifelike speech.
- **TavusVideoService** for an animated avatar persona.
- A custom **ConversationCoach** that injects stimuli and native-language nudges when the learner hesitates or falls silent.

> ✅ This repository only contains the server/agent implementation. A client (web, mobile, etc.) must provide the `webrtc_connection` and Tavus session objects at runtime.

## Project layout

```
.
├── pyproject.toml            # Package + tooling metadata
├── requirements.txt          # Convenient dependency pinning
├── src/langlearn_agent/
│   ├── __init__.py
│   ├── config.py             # Dataclasses for build + session inputs
│   ├── coaching.py           # ConversationCoach frame processor
│   ├── pipeline.py           # LanguageLearningAgent builder
│   ├── prompts.py            # System + scenario prompt helpers
│   └── runtime.py            # build_agent_from_env helper
└── README.md
```

## Prerequisites

1. Python 3.10+
2. Access credentials for all external services:
   - `DEEPGRAM_API_KEY`
   - `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`
   - `OPEN_ROUTER_API_KEY`
   - `TAVUS_API_KEY` and `TAVUS_REPLICA_ID`
3. Active WebRTC + Tavus sessions created by your client application (outside this repo).

Install dependencies with either `pip install -r requirements.txt` or `pip install .`.

## Quickstart

```python
import asyncio
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
# Whatever SDK you use to negotiate Tavus sessions:
from tavus_sdk import TavusSession

from langlearn_agent.runtime import build_agent_from_env


async def main():
    # Replace the ellipses with your actual SmallWeb connection details. Pipecat's
    # SmallWebRTCConnection just needs the signalling URL + auth token that your
    # client transport already knows about.
    # Example kwargs; consult pipecat.transports.smallwebrtc.connection docs
    # for the exact constructor signature your deployment expects.
    webrtc_connection = SmallWebRTCConnection(
        host="wss://your-smallweb-edge",
        token="signed-client-token",
        room_id="room-123",
    )

    tavus_session = TavusSession(...)

    bundle = build_agent_from_env(
        scenario_prompt="Simulate a conversation with a waiter at a restaurant",
        native_language="English",
        target_language="Spanish",
        learner_name="Jules",
        proficiency="intermediate",
        supplemental_vocab=("la cuenta", "propina"),
        webrtc_connection=webrtc_connection,
        tavus_session=tavus_session,
    )

    await bundle.pipeline.run()  # or schedule via your orchestration layer


asyncio.run(main())
```

`build_agent_from_env` calls `AgentBuildConfig.from_env`, which automatically loads `.env` (via `python-dotenv`). Pass `dotenv_path="path/to/.env"` if your credentials live elsewhere.

## FastAPI SmallWeb bridge

For an end-to-end demo you can run the included FastAPI app (`langlearn_agent.server`) which exposes:

- `POST /start` – mimics Pipecat Cloud’s session bootstrap.
- `POST /api/offer` / `PATCH /api/offer` – WebRTC signaling endpoints handled by `SmallWebRTCRequestHandler`.
- `/prebuilt` – Pipecat’s SmallWeb prebuilt UI for manual testing.

Each accepted WebRTC connection spins up a pipeline via `run_bot` in `langlearn_agent.bot`, which:

1. Reads scenario + learner metadata from the `LANG_AGENT_*` env vars (optional, see below).
2. Creates a Tavus session via `TAVUS_API_KEY`/`TAVUS_REPLICA_ID`.
3. Calls `build_agent_from_env(...)` and runs the pipeline until the call ends.

Run the server with:

```
python -m langlearn_agent.server --host 0.0.0.0 --port 7860
```

You can tune the default session via environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `LANG_AGENT_SCENARIO` | Prompt for the role play | “Simulate a conversation with a waiter…” |
| `LANG_AGENT_NATIVE` / `LANG_AGENT_TARGET` | Native + target languages | `English` / `Spanish` |
| `LANG_AGENT_LEARNER_NAME` | Friendly learner name | `Learner` |
| `LANG_AGENT_PROFICIENCY` | Proficiency tag fed into prompts | `beginner` |
| `LANG_AGENT_VOCAB` | Comma-separated list of vocab words to recycle | _(empty)_ |
| `LANG_AGENT_DELIVER_STIMULI` | `true`/`false` toggle for markdown stimuli | `true` |
| `LANG_AGENT_STIMULUS_CHANNEL` | Label describing how stimuli should be rendered | `markdown` |

Additional integration knobs:

- `TAVUS_API_BASE_URL` (optional) – override Tavus REST base URL if you’re pointing at a staging cluster.
- `SMALLWEB_HOST`, `SMALLWEB_TOKEN`, `SMALLWEB_ROOM` – used by `python -m langlearn_agent.bot` for quick CLI smoke-tests without the FastAPI server.

At runtime the pipeline stages are:

1. `SmallWebRTCTransport.input()` captures microphone audio, performs Silero VAD, and forwards frames.
2. `DeepgramFluxSTTService` streams transcripts into the pipeline.
3. `TranscriptProcessor.user()` turns raw STT frames into structured messages.
4. `ConversationCoach` injects system cues whenever the learner stalls so the LLM briefly reverts to the native language before resuming the target tongue.
5. `LLMContextAggregatorPair.user()` maintains OpenAI-format context, seeding it with stimuli built from the learner prompt.
6. `OpenAILLMService` (via OpenRouter) generates the assistant reply within the scenario.
7. `ElevenLabsTTSService` vocalizes the response.
8. `TavusVideoService` renders the avatar video stream that mouths the TTS audio.
9. `SmallWebRTCTransport.output()` ships audio/video back to the learner.
10. `TranscriptProcessor.assistant_tts()` and `context_aggregator.assistant()` keep conversation state aligned.

## Customization notes

- **Stimuli** – The system prompt logic automatically instructs the LLM to emit concise markdown stimuli (menus, tickets, etc.). Set `deliver_stimuli=False` when building a `LanguageLearningSession` to disable this behavior.
- **Native-language nudges** – Tune `AgentBuildConfig.stuck_timeout_seconds` and `hesitant_utterance_tokens` to control when the `ConversationCoach` interrupts with native-language assistance.
- **Model selection** – Override `AgentBuildConfig.openrouter_model` with any OpenRouter-compatible model ID.

## Next steps

- Implement the corresponding client transport (SmallWeb or Pipecat client SDK) that acquires `webrtc_connection` and Tavus sessions.
- Add persistence/logging for transcripts via `PipelineBundle.transcript`.
- Extend `ConversationCoach` with richer heuristics (e.g., speech-energy thresholds, proficiency-aware hints).
