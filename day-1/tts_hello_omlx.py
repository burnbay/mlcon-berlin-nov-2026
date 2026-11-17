"""The oMLX twin of tts_hello.py: the smallest possible TTS call, Chatterbox-Turbo again, but served by oMLX
over its OpenAI-compatible /v1/audio/speech instead of running mlx-audio in this script.
Writes day-1/output/output_omlx.wav and plays it. stt_transcribe_omlx.py turns it back into text.

    python day-1/tts_hello_omlx.py ["Some other text"]

Chatterbox's built-in voice speaks, unless a cloning demo ran first: oMLX keeps the last cloned voice until
Chatterbox is unloaded (on oMLX's admin page, or by restarting oMLX).

Needs oMLX running on :8000 with mlx-community/chatterbox-turbo-fp16 in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv sounddevice numpy`.
"""
import io
import os
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

HERE = Path(__file__).parent

if __name__ == "__main__":
    wav = client.audio.speech.create(
        input=sys.argv[1] if len(sys.argv) > 1 else "Hello World",
        model="mlx-community--chatterbox-turbo-fp16",   # oMLX names Hugging Face cache models "org--repo"
        voice="default",   # the OpenAI SDK insists on a voice; Chatterbox has no named voices, so oMLX ignores it
        response_format="wav",
    ).content
    out = HERE / "output" / "output_omlx.wav"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(wav)
    print(f"Saved {out}")
    with wave.open(io.BytesIO(wav)) as w:   # oMLX sends 16-bit mono WAV; play it like mlx-audio's play=True does
        sd.play(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate(), blocking=True)
