import os
import multiprocessing
import struct
from flask import Flask, Response, request
from orpheus_tts import OrpheusModel
from vllm import AsyncEngineArgs, AsyncLLMEngine


# =====================================================
# 🧩 Monkeypatch ép cấu hình VLLM (giữ engine persistent)
# =====================================================

def patched_setup_engine(self):
    """Custom setup giữ engine persistent, tương thích mọi version vLLM."""
    engine_args = AsyncEngineArgs(
        model=self.model_name,
        dtype="float16",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_chunked_prefill=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=True,
        disable_log_stats=True,
        enforce_eager=False,
    )

    # Tạo engine và giữ tham chiếu lại (ngăn bị GC cleanup)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    self._persistent_engine = engine
    print("✅ Persistent AsyncLLMEngine initialized and retained.")
    return engine


# Ghi đè method gốc tạm thời
OrpheusModel._setup_engine = patched_setup_engine

# =====================================================
# Flask App
# =====================================================
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    app = Flask(__name__)

    engine = OrpheusModel(
        model_name="canopylabs/orpheus-tts-0.1-finetune-prod"
    )

    print("✅ Orpheus engine loaded with max_model_len=32768")


    def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
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
            data_size
        )
        return header


    @app.route("/tts", methods=["GET"])
    def tts():
        prompt = request.args.get("prompt", "Hey there, looks like you forgot to provide a prompt!")

        def generate_audio_stream():
            yield create_wav_header()
            syn_tokens = engine.generate_speech(
                prompt=prompt,
                voice="tara",
                repetition_penalty=1.1,
                stop_token_ids=[128258],
                max_tokens=2000,
                temperature=0.4,
                top_p=0.9
            )
            for chunk in syn_tokens:
                yield chunk

        return Response(generate_audio_stream(), mimetype="audio/wav")


    import openai

    # Set your API key (có thể đặt qua biến môi trường)
    openai.api_key = os.getenv("OPENAI_API_KEY")


    @app.route("/chat_tts", methods=["GET"])
    def chat_tts():
        user_prompt = request.args.get("prompt", "Hello there!")

        # Step 1: Gọi OpenAI LLM (không cần streaming)
        try:
            llm_response = openai.chat.completions.create(
                model="gpt-4o-mini",  # hoặc gpt-4o nếu bạn có quyền
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that speaks naturally."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            reply_text = llm_response.choices[0].message.content.strip()
            print(f"🤖 LLM replied: {reply_text}")
        except Exception as e:
            reply_text = "Sorry, I could not process your question."
            print(f"⚠️ LLM error: {e}")

        # Step 2: Sinh audio từ text trả về
        def generate_audio_stream():
            yield create_wav_header()
            syn_tokens = engine.generate_speech(
                prompt=reply_text,
                voice="tara",
                repetition_penalty=1.1,
                stop_token_ids=[128258],
                max_tokens=2000,
                temperature=0.4,
                top_p=0.9
            )
            for chunk in syn_tokens:
                yield chunk

        return Response(generate_audio_stream(), mimetype="audio/wav")


    app.run(host="0.0.0.0", port=8080, threaded=True)