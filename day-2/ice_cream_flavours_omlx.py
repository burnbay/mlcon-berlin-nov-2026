"""The oMLX twin of ice_cream_flavours.py: same menu photo and prompt, thinking still off, still one raw HTTP
POST, but to oMLX's OpenAI-compatible endpoint instead of Ollama's /api/chat.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install requests python-dotenv`.
"""
import base64
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")

HERE = Path(__file__).parent

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    payload = {
        "model": "mlx-community--Qwen3.5-4B-MLX-4bit",
        "messages": [{
                "role": "system",
                "content": "You are a helpful visual assistant, give detailed and accurate answers.",
            },{
                "role": "user", "content": [
                    {"type": "text", "text": "List the ice cream flavours shown on the menu in this image."},
                    # OpenAI-style APIs take the image as a data URL inside the message content
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + encode_image_to_base64(HERE / "data" / "IMG_3319.jpeg")}}, ], },],
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7, "max_tokens": 1000,
    }

    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        data=json.dumps(payload),
        timeout=120,
    )
    response.raise_for_status()
    print(response.json()["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
