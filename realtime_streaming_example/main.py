"""
Realtime Voice Assistant pipeline
using OpenAI STT + LLM + Orpheus TTS (patched VLLM config).
"""

import io
import json
import os
import struct
import uuid
import multiprocessing
from typing import Generator, List, Optional

from flask import Flask, Response, jsonify, request

# =====================================================
# ⚙️ Patch cấu hình VLLM TRƯỚC khi khởi tạo OrpheusModel
# =====================================================
from orpheus_tts import OrpheusModel
from vllm import AsyncEngineArgs, AsyncLLMEngine


def patched_setup_engine(self):
    """Monkeypatch: ép cấu hình VLLM cho vừa VRAM"""
    engine_args = AsyncEngineArgs(
        model=self.model_name,
        dtype=self.dtype,
        max_model_len=32768,          # ✅ giảm context length
        gpu_memory_utilization=0.9,   # ✅ phù hợp GPU 24GB (A4000, A10G)
        trust_remote_code=True,
        disable_log_stats=True,
        enforce_eager=False,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


# Ghi đè method gốc tạm thời
OrpheusModel._setup_engine = patched_setup_engine


# =====================================================
# 🧠 OpenAI client setup (for STT + LLM)
# =====================================================
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

openai_client: Optional[OpenAI] = None

OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv(
    "VOICE_ASSISTANT_SYSTEM_PROMPT",
    "You are a helpful and friendly voice assistant. Keep replies short and natural.",
)


def require_openai_client() -> OpenAI:
    """Lấy hoặc khởi tạo OpenAI client"""
    if OpenAI is None:
        raise RuntimeError("Missing openai package. Run: pip install openai")

    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
        openai_client = OpenAI(api_key=api_key)
    return openai_client


# =====================================================
# 🗣️ Orpheus TTS Engine
# =====================================================
_engine: Optional[OrpheusModel] = None
_ENGINE_MODEL_NAME = os.getenv(
    "ORPHEUS_MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod"
)


def get_tts_engine() -> OrpheusModel:
    """Lazy-load TTS engine (singleton)"""
    global _engine
    if _engine is None:
        print("🔄 Initializing Orpheus TTS engine ...")
        _engine = OrpheusModel(model_name=_ENGINE_MODEL_NAME)
    return _engine


def reset_tts_engine() -> None:
    """Reset global engine on failure"""
    global _engine
    _engine = None


def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1) -> bytes:
    """Chuẩn WAV header cho streaming"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0
    return struct.pack(
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


def generate_tts_stream(text: str, voice="tara") -> Generator[bytes, None, None]:
    """Sinh luồng WAV audio từ văn bản (streaming)"""
    attempts = 0
    while attempts < 2:
        engine = get_tts_engine()
        request_id = f"req-{uuid.uuid4()}"

        try:
            yield create_wav_header()
            for chunk in engine.generate_speech(
                prompt=text,
                voice=voice,
                repetition_penalty=1.1,
                stop_token_ids=[128258],
                max_tokens=2000,
                temperature=0.4,
                top_p=0.9,
                request_id=request_id,
            ):
                if chunk:
                    yield chunk
            return
        except Exception as e:
            print(f"[TTS] Error: {e} → resetting engine...")
            reset_tts_engine()
            attempts += 1
    raise RuntimeError("TTS engine unavailable")


# =====================================================
# 🎙️ Speech-to-Text (Whisper)
# =====================================================
def transcribe_audio(audio_file) -> str:
    client = require_openai_client()
    audio_bytes = audio_file.read()
    if not audio_bytes:
        raise ValueError("Uploaded audio file is empty")

    buf = io.BytesIO(audio_bytes)
    buf.name = audio_file.filename or "audio.wav"

    result = client.audio.transcriptions.create(
        model=OPENAI_STT_MODEL,
        file=buf,
    )
    return result.text.strip()


# =====================================================
# 💬 Chat LLM
# =====================================================
def call_llm(prompt: str, history: Optional[List[dict]] = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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


# =====================================================
# 🚀 Flask API
# =====================================================
app = Flask(__name__)


@app.before_first_request
def warmup_tts():
    """Warm-up TTS engine khi khởi động"""
    try:
        get_tts_engine()
        print("✅ Orpheus TTS engine ready!")
    except Exception as e:
        print(f"⚠️ Warm-up failed: {e}")


@app.route("/tts", methods=["GET"])
def tts():
    prompt = request.args.get("prompt", "Hello Én! This is Orpheus speaking.")
    voice = request.args.get("voice", "tara")
    return Response(generate_tts_stream(prompt, voice=voice), mimetype="audio/wav")


@app.route("/stt", methods=["POST"])
def stt():
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' in form-data"}), 400
    audio_file = request.files["audio"]
    try:
        text = transcribe_audio(audio_file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"transcript": text})


@app.route("/llm", methods=["POST"])
def llm():
    payload = request.get_json(silent=True) or {}
    prompt = payload.get("prompt")
    history = payload.get("history")
    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400
    if history is not None and not isinstance(history, list):
        return jsonify({"error": "'history' must be a list"}), 400

    try:
        text = call_llm(prompt, history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"response": text})


@app.route("/voice-assistant", methods=["POST"])
def voice_assistant():
    """Full pipeline: STT → LLM → TTS"""
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' file"}), 400

    history_raw = request.form.get("history")
    history = None
    if history_raw:
        try:
            history = json.loads(history_raw)
            if not isinstance(history, list):
                raise ValueError
        except ValueError:
            return jsonify({"error": "Invalid 'history' payload"}), 400

    audio_file = request.files["audio"]
    try:
        transcript = transcribe_audio(audio_file)
        assistant_text = call_llm(transcript, history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    headers = {
        "X-Transcript": sanitize_header_value(transcript),
        "X-Assistant-Text": sanitize_header_value(assistant_text),
    }
    return Response(generate_tts_stream(assistant_text), mimetype="audio/wav", headers=headers)

@app.route("/voice-assistant-stream", methods=["POST"])
def voice_assistant_stream():
    """Realtime: STT → LLM → stream TTS chunks"""
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio'"}), 400

    audio_file = request.files["audio"]
    history_raw = request.form.get("history")
    history = None
    if history_raw:
        try:
            history = json.loads(history_raw)
        except Exception:
            history = None

    try:
        transcript = transcribe_audio(audio_file)
        assistant_text = call_llm(transcript, history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    def generate_stream():
        # gửi header WAV
        yield create_wav_header()
        for chunk in generate_tts_stream(assistant_text):
            if chunk:
                yield chunk

    headers = {
        "X-Transcript": sanitize_header_value(transcript),
        "X-Assistant-Text": sanitize_header_value(assistant_text),
    }
    return Response(generate_stream(), mimetype="audio/wav", headers=headers)


# =====================================================
# 🏁 Run Flask
# =====================================================
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    app.run(host="0.0.0.0", port=8080, threaded=True)
