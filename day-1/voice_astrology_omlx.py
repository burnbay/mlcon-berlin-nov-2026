"""The oMLX twin of voice_astrology.py: tell Maude your name and star sign out loud and she speaks your
horoscope back. Whisper (speech-to-text), Qwen (Maude) and the voice all run on this Mac under oMLX
instead of on the class Spark.

    python voice_astrology_omlx.py            (speak into the microphone)
    python voice_astrology_omlx.py some.wav   (use a recording instead)

Needs oMLX running on :8000 with mlx-community/whisper-large-v3-turbo-asr-4bit, mlx-community/Qwen3.5-4B-MLX-4bit
and mlx-community/chatterbox-turbo-fp16 in the Hugging Face cache, and OMLX_API_KEY in the environment or the
repo-root .env. Needs `pip install openai python-dotenv sounddevice numpy`.
"""
import base64
import io
import os
import sys
import wave
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
HERE = Path(__file__).parent

# Everything runs on oMLX on this Mac: Whisper (speech-to-text), Qwen (Maude) and the voice.
api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)
WHISPER = "mlx-community--whisper-large-v3-turbo-asr-4bit"   # oMLX names Hugging Face cache models "org--repo"
MAUDE = "mlx-community--Qwen3.5-4B-MLX-4bit"   # the Spark ran qwen3.6-35b; Qwen3.8-27B-oQ4e-mtp is sharper but slower
CHATTERBOX = "mlx-community--chatterbox-turbo-fp16"   # not 8-bit: it lacks the speaker-encoder weights cloning needs

# ---- things to change -----------------------------------------------------------
VOICE_CLIP = None        # e.g. HERE / "data" / "roosevelt-voice-sample.wav" to clone a voice
                         # None: Chatterbox's own built-in voice (there is no Kokoro on oMLX here), unless a
                         # voice was cloned earlier: oMLX keeps that one until Chatterbox is unloaded
SYSTEM_PROMPT = ("You are Maude, an AI astrologer. Give a short, positive and optimistic horoscope for "
                 "tomorrow in three spoken sentences: no lists, no markdown, no emojis.")
RATE, SILENCE_RMS, SILENCE_SECS, MAX_SECS = 16000, 0.01, 1.2, 15
# ---------------------------------------------------------------------------------


@cache
def clone(clip):
    """The voice to imitate, plus its transcript: oMLX insists on one (Chatterbox only listens to the audio)."""
    with open(clip, "rb") as f:
        ref_text = client.audio.transcriptions.create(model=WHISPER, file=f).text
    return {"ref_audio": base64.b64encode(Path(clip).read_bytes()).decode(), "ref_text": ref_text}


def speak(text):
    # voice: the OpenAI SDK insists on one; Chatterbox has no named voices, so oMLX ignores it
    wav = client.audio.speech.create(model=CHATTERBOX, voice="default", input=text, response_format="wav",
                                     extra_body=clone(str(VOICE_CLIP)) if VOICE_CLIP else None).content
    with wave.open(io.BytesIO(wav)) as w:
        sd.play(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate(), blocking=True)


def listen():
    """Record until a pause and return it as a WAV file object."""
    chunks, heard, quiet = [], False, 0
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32") as mic:
        for _ in range(MAX_SECS * 10):
            chunk, _ = mic.read(RATE // 10)
            chunks.append(chunk)
            loud = np.sqrt(np.mean(chunk ** 2)) > SILENCE_RMS
            heard |= loud
            quiet = 0 if loud else quiet + 1
            if heard and quiet >= SILENCE_SECS * 10:
                break
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.concatenate(chunks) * 32767).astype(np.int16).tobytes())
    buf.name = "you.wav"
    buf.seek(0)
    return buf


if __name__ == "__main__":
    speak("Hello, I'm Maude. Tell me your name and your star sign.")
    # python voice_astrology_omlx.py some.wav  uses a recording instead of the microphone
    audio = open(sys.argv[1], "rb") if len(sys.argv) > 1 else listen()
    heard = client.audio.transcriptions.create(model=WHISPER, file=audio, language="en").text
    print("You:", heard)

    horoscope = client.chat.completions.create(
        model=MAUDE,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"{heard} Today's date is {date.today():%A, %d %B %Y}."}],
        temperature=0.8, max_tokens=200,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).choices[0].message.content
    print("Maude:", horoscope)
    speak(horoscope)
