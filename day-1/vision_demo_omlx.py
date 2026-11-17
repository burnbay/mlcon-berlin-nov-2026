"""The oMLX twin of vision_demo_ollama.py: same prompt, a local image.jpg described and placed, thinking still
off, but served by oMLX over its OpenAI-compatible API instead of Ollama. Drop in any image as image.jpg in
the folder you run it from.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv`.
"""
import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

with open("image.jpg", "rb") as f:
    image_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()   # OpenAI-style APIs take images as data URLs

response = client.chat.completions.create(
    # The Ollama version uses qwen3.6:27b; "Qwen3.8-27B-oQ4e-mtp" is the sharper but slower option on oMLX
    model="mlx-community--Qwen3.5-4B-MLX-4bit",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image and guess where it might have been taken."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
