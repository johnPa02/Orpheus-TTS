"""Realtime voice assistant pipeline using Orpheus TTS, OpenAI STT, and LLM."""

import io
import json
import os
import struct
from typing import Generator, List, Optional

from flask import Flask, Response, jsonify, request

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - allows app import without OpenAI
    OpenAI = None  # type: ignore[assignment]

from orpheus_tts import OrpheusModel

app = Flask(__name__)
engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
openai_client: Optional[OpenAI] = None

OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv(
    "VOICE_ASSISTANT_SYSTEM_PROMPT",
    "You are a helpful voice assistant. Keep responses concise and conversational.",
)


def require_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Install it with 'pip install openai'."
        )

    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        openai_client = OpenAI(api_key=api_key)
    return openai_client


def create_wav_header(sample_rate: int = 24000, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size,
    )
    return header


def generate_tts_stream(text: str) -> Generator[bytes, None, None]:
    yield create_wav_header()

    syn_tokens = engine.generate_speech(
        prompt=text,
        voice="tara",
        repetition_penalty=1.1,
        stop_token_ids=[128258],
        max_tokens=2000,
        temperature=0.4,
        top_p=0.9,
    )
    for chunk in syn_tokens:
        yield chunk


def transcribe_audio(audio_file) -> str:
    client = require_openai_client()
    audio_bytes = audio_file.read()
    if not audio_bytes:
        raise ValueError("Uploaded audio file is empty")

    buffer = io.BytesIO(audio_bytes)
    buffer.name = audio_file.filename or "audio.wav"

    transcription = client.audio.transcriptions.create(
        model=OPENAI_STT_MODEL,
        file=buffer,
    )
    return transcription.text


def call_llm(prompt: str, history: Optional[List[dict]] = None) -> str:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    client = require_openai_client()
    completion = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()


def sanitize_header_value(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.replace("\r", " ").replace("\n", " ")[:1000]


@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
    return Response(generate_tts_stream(prompt), mimetype='audio/wav')


@app.route('/stt', methods=['POST'])
def stt():
    if 'audio' not in request.files:
        return jsonify({'error': "Missing 'audio' file in form-data"}), 400

    audio_file = request.files['audio']
    try:
        transcript = transcribe_audio(audio_file)
    except Exception as exc:  # noqa: BLE001 - return controlled error to client
        return jsonify({'error': str(exc)}), 500

    return jsonify({'transcript': transcript})


@app.route('/llm', methods=['POST'])
def llm():
    payload = request.get_json(silent=True) or {}
    prompt = payload.get('prompt')
    if not prompt:
        return jsonify({'error': "Missing 'prompt' in JSON body"}), 400

    history = payload.get('history')
    if history is not None and not isinstance(history, list):
        return jsonify({'error': "'history' must be a list of messages"}), 400

    try:
        response_text = call_llm(prompt, history)
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500

    return jsonify({'response': response_text})


@app.route('/voice-assistant', methods=['POST'])
def voice_assistant():
    if 'audio' not in request.files:
        return jsonify({'error': "Missing 'audio' file in form-data"}), 400

    history_raw = request.form.get('history')
    history: Optional[List[dict]] = None
    if history_raw:
        try:
            history = json.loads(history_raw)
            if not isinstance(history, list):
                raise ValueError
        except ValueError:
            return jsonify({'error': "Invalid 'history' payload; expected JSON list"}), 400

    audio_file = request.files['audio']
    try:
        transcript = transcribe_audio(audio_file)
        assistant_text = call_llm(transcript, history)
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500

    headers = {
        'X-Transcript': sanitize_header_value(transcript),
        'X-Assistant-Text': sanitize_header_value(assistant_text),
    }

    return Response(generate_tts_stream(assistant_text), mimetype='audio/wav', headers=headers)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
