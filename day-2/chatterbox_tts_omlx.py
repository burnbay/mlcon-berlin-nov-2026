"""The oMLX twin of chatterbox_tts.py (MLX) and chatterbox_turbo_tts.py (PyTorch): the same Chatterbox-Turbo
text-to-speech with voice cloning, but served by oMLX over its OpenAI-compatible API, so nothing loads in this
script. Writes day-2/output/demo_omlx.wav and plays it.

Needs oMLX running on :8000 with mlx-community/chatterbox-turbo-fp16 and mlx-community/whisper-large-v3-turbo-asr-4bit
in the Hugging Face cache, and OMLX_API_KEY in the environment or the repo-root .env.
Needs `pip install openai python-dotenv sounddevice numpy`.
"""
import base64
import io
import os
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
VOICE = HERE / "data" / "morgan-freeman-voice-sample.wav"   # the voice to clone: try your own 10 s recording
TEXT = ("In the autumn of 2026, a quiet gathering took place in Berlin. They called it MLCon Berlin. "
        "And on that Tuesday morning, as the light came through the windows, everyone in the room "
        "understood that something remarkable was about to unfold.")

# oMLX wants the voice sample's transcript alongside it (Chatterbox itself only listens to the audio),
# so Whisper on oMLX writes one first
with VOICE.open("rb") as f:
    ref_text = client.audio.transcriptions.create(model="mlx-community--whisper-large-v3-turbo-asr-4bit", file=f).text

wav = client.audio.speech.create(
    # fp16 like chatterbox_tts.py, not 8-bit: the quantized repos lack Chatterbox's speaker-encoder weights,
    # so their clones lose the voice
    model="mlx-community--chatterbox-turbo-fp16",   # oMLX names Hugging Face cache models "org--repo"
    voice="default",   # the OpenAI SDK insists on a voice; Chatterbox has no named voices, so oMLX ignores it
    input=TEXT,
    response_format="wav",
    extra_body={"ref_audio": base64.b64encode(VOICE.read_bytes()).decode(), "ref_text": ref_text},
).content

out = HERE / "output" / "demo_omlx.wav"
out.parent.mkdir(exist_ok=True)
out.write_bytes(wav)
print(f"Saved {out}")

with wave.open(io.BytesIO(wav)) as w:   # oMLX sends 16-bit mono WAV; play it like play=True in chatterbox_tts.py
    sd.play(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate(), blocking=True)
