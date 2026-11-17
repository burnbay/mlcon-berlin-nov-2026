"""The oMLX twin of stt_transcribe.py: transcribe a recording with Whisper large-v3-turbo again, but served by
oMLX over its OpenAI-compatible /v1/audio/transcriptions instead of running mlx-audio in this script.
By default it reads the file tts_hello_omlx.py wrote. Tip: pass language="en" when you know it,
and don't send silence (Whisper hears it as "Thank you.").

    python day-1/stt_transcribe_omlx.py [some.wav]

Needs oMLX running on :8000 with mlx-community/whisper-large-v3-turbo-asr-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv`.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

HERE = Path(__file__).parent

if __name__ == "__main__":
    with open(sys.argv[1] if len(sys.argv) > 1 else HERE / "output" / "output_omlx.wav", "rb") as audio:
        result = client.audio.transcriptions.create(
            model="mlx-community--whisper-large-v3-turbo-asr-4bit",   # oMLX names Hugging Face cache models "org--repo"
            file=audio,
        )
    print(result.text)
