# Realtime Streaming Example

This example exposes a Flask API that wires together a full speech pipeline:

1. **STT** — OpenAI Whisper via `audio.transcriptions.create`
2. **LLM** — OpenAI Chat Completions for fast response generation
3. **TTS** — Orpheus realtime streaming (existing implementation)

## Endpoints

| Route | Method | Description |
| --- | --- | --- |
| `/tts?prompt=...` | GET | Streams Orpheus speech for the supplied text prompt (existing behaviour). |
| `/stt` | POST | Accepts multipart form-data with an `audio` file and returns the Whisper transcript. |
| `/llm` | POST | Accepts JSON `{"prompt": "...", "history": [...]}` and returns an assistant reply. |
| `/voice-assistant` | POST | Accepts multipart form-data `{ audio, history? }`, performs STT → LLM, then streams the TTS response. Response headers expose the transcript (`X-Transcript`) and the LLM output (`X-Assistant-Text`). |

## Configuration

```bash
export OPENAI_API_KEY=sk-...
# Optional overrides
export OPENAI_STT_MODEL=whisper-1
export OPENAI_CHAT_MODEL=gpt-4o-mini
export VOICE_ASSISTANT_SYSTEM_PROMPT="You are a helpful voice assistant."
# swap Orpheus model if you have another checkpoint
export ORPHEUS_MODEL_NAME=canopylabs/orpheus-tts-0.1-finetune-prod
```

Install the dependencies (adjust as needed for your environment):

```bash
pip install flask openai
# plus Orpheus from the repo root
pip install -e .
```

Run the server:

```bash
python realtime_streaming_example/main.py
```

## Frontend Demo

`client.html` contains a minimal UI to exercise the endpoints:

- **TTS Only** — send a text prompt directly to `/tts` and stream the audio response.
- **Voice Assistant** — record microphone input, POST to `/voice-assistant`, and play the streamed reply while showing the transcript and generated text.

Update `baseUrl` at the top of the script to point to the server before opening the file in a browser.
