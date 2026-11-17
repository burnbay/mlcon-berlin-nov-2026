"""The oMLX twin of test_lmstudio_vision.py: the smallest possible vision call (one image, one sentence back),
but served by oMLX over its OpenAI-compatible API instead of the LM Studio SDK.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv`.
"""
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

HERE = Path(__file__).parent
IMAGE = HERE / "data" / "RTS1UI9-1024x659.jpg"

model = "mlx-community--Qwen3.5-4B-MLX-4bit"
image = "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode()   # instead of lms.prepare_image

# No settings, like the LM Studio version: the model's default thinking applies, and oMLX keeps it out of .content
result = client.chat.completions.create(model=model, messages=[
    {"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."},
                                 {"type": "image_url", "image_url": {"url": image}}]},
])
print(result.choices[0].message.content)
