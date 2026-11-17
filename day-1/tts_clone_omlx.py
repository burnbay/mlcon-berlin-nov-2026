"""The oMLX twin of tts_clone.py: clone a voice from a short recording with Chatterbox-Turbo again, but served
by oMLX over its OpenAI-compatible API instead of running mlx-audio in this script.
The reference is day-1/data/morgan-freeman-voice-sample.wav; pass any 5-15 s WAV to clone another voice
(day-1/data/roosevelt-voice-sample.wav is Theodore Roosevelt in 1912, public domain).
Only clone a voice with its owner's consent. Writes day-1/output/morgan_omlx.wav and plays it.

    python day-1/tts_clone_omlx.py [reference.wav]

Needs oMLX running on :8000 with mlx-community/chatterbox-turbo-fp16 and mlx-community/whisper-large-v3-turbo-asr-4bit
in the Hugging Face cache, and OMLX_API_KEY in the environment or the repo-root .env.
Needs `pip install openai python-dotenv sounddevice numpy`.
"""
import base64
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
    ref_audio = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "data" / "morgan-freeman-voice-sample.wav")
    # oMLX wants the reference clip's transcript alongside it (Chatterbox itself only listens to the audio),
    # so Whisper on oMLX writes one first: speech-to-text feeding text-to-speech
    with ref_audio.open("rb") as f:
        ref_text = client.audio.transcriptions.create(model="mlx-community--whisper-large-v3-turbo-asr-4bit",
                                                      file=f).text
    wav = client.audio.speech.create(
        input="""Hello, fellow travellers.
        You're here to learn how machines find their voice,
        and how they listen. It's a curious thing, teaching
        silicon to speak. But every great journey begins
        with a single, well-chosen word. Welcome.""",
        # fp16, not 8-bit: the quantized repos lack Chatterbox's speaker-encoder weights, so their clones lose the voice
        model="mlx-community--chatterbox-turbo-fp16",   # oMLX names Hugging Face cache models "org--repo"
        voice="default",   # the OpenAI SDK insists on a voice; Chatterbox has no named voices, so oMLX ignores it
        response_format="wav",
        extra_body={"ref_audio": base64.b64encode(ref_audio.read_bytes()).decode(),   # the voice to clone
                    "ref_text": ref_text},
    ).content
    out = HERE / "output" / "morgan_omlx.wav"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(wav)
    print(f"Saved {out}")
    with wave.open(io.BytesIO(wav)) as w:   # oMLX sends 16-bit mono WAV; play it like mlx-audio's play=True does
        sd.play(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate(), blocking=True)
