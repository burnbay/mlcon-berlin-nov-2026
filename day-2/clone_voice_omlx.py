"""The oMLX twin of clone_voice.py: Chatterbox speaks any text in the voice of a short sample, but served by
oMLX on this Mac over its OpenAI-compatible API instead of on the class Spark.

Plays the result out loud and saves it to day-2/output/.
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
WHISPER = "mlx-community--whisper-large-v3-turbo-asr-4bit"   # oMLX names Hugging Face cache models "org--repo"
CHATTERBOX = "mlx-community--chatterbox-turbo-fp16"   # not 8-bit: it lacks the speaker-encoder weights cloning needs

VOICE = HERE / "data" / "roosevelt-voice-sample.wav"   # 5-15 s of one speaker; try morgan-freeman-voice-sample.wav, or your own
TEXT = "Guten Morgen, Berlin. Today we build applications with open models that run in this very room."
LANGUAGE = "en"          # the language of TEXT. Chatterbox-Turbo speaks English only: oMLX accepts a language
                         # but this model ignores it (the Spark ran the 23-language Chatterbox)
TEMPERATURE = 0.8        # randomness: every run gives a slightly different reading
# No EXAGGERATION / CFG_WEIGHT: Chatterbox-Turbo has neither knob, and oMLX doesn't pass them on


def clone_voice(text, sample):
    """Send the text and the voice sample, get back WAV bytes."""
    # oMLX wants the sample's transcript alongside it (Chatterbox itself only listens to the audio):
    # Whisper on oMLX writes one first
    with sample.open("rb") as f:
        transcript = client.audio.transcriptions.create(model=WHISPER, file=f).text
    response = client.audio.speech.create(
        model=CHATTERBOX,
        voice="default",                    # the OpenAI SDK insists on one; Chatterbox has no named voices
        input=text,
        response_format="wav",
        extra_body={
            "ref_audio": base64.b64encode(sample.read_bytes()).decode(),   # the voice to imitate
            "ref_text": transcript,                                        # what the sample says
            "language": LANGUAGE,
            "temperature": TEMPERATURE,
        },
    )
    return response.content


def play(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
        sd.play(audio, w.getframerate())
        sd.wait()


if __name__ == "__main__":
    print(f"Cloning the voice in {VOICE.name} ...")
    wav_bytes = clone_voice(TEXT, VOICE)

    out = HERE / "output" / f"cloned_{VOICE.stem}_omlx.wav"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(wav_bytes)
    print(f"Saved {out}")

    print("Playing ...")
    play(wav_bytes)
