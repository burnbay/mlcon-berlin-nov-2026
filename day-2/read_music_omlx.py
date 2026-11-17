"""The oMLX twin of read_music.py: same score, same key-signature prompt, thinking still off, but served by oMLX
over its OpenAI-compatible API instead of Ollama.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv`.
"""
import base64
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

HERE = Path(__file__).parent

MODELS: List[str] = [
    "mlx-community--Qwen3.5-4B-MLX-4bit",   # oMLX names Hugging Face cache models "org--repo"
]
IMAGE_PATH: str = str(HERE / "data" / "IMG_9.jpg")


def get_music_prompt(model_name: str) -> str:
    return "Read the key of the music score in this image. What is the musical key signature?"


def image_message(prompt: str, image_path: str) -> List[Dict[str, Any]]:
    # Ollama takes the file path; OpenAI-style APIs take the image as a base64 data URL
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    return [{
        'role': 'user',
        'content': [{'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{b64}"}}]
    }]


def warmup_model(model_name: str, image_path: str) -> None:
    print(f"Warming up model: {model_name}...")

    try:
        dummy_prompt = "What do you see in this image?"

        client.chat.completions.create(
            model=model_name,
            messages=image_message(dummy_prompt, image_path),
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        print(f"Model {model_name} loaded and ready")
    except Exception as e:
        print(f"Warmup failed for {model_name}: {e}")


def read_music_key(model_name: str, image_path: str):
    prompt = get_music_prompt(model_name)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=image_message(prompt, image_path),
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    inference_time = time.time() - start_time

    return response.choices[0].message.content, inference_time


def print_header(title: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_result(model_name: str, text: str, inference_time: float, width: int = 60) -> None:
    print(f"\nModel: {model_name}")
    print("-" * width)
    print(text)
    print(f"\nInference time: {inference_time:.2f}s")
    print("=" * width)


def main() -> None:
    print_header("MUSIC SCORE KEY READING DEMO")
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {', '.join(MODELS)}")

    results: Dict[str, Any] = {}

    for model in MODELS:
        try:
            warmup_model(model, IMAGE_PATH)

            text, inference_time = read_music_key(model, IMAGE_PATH)
            print_result(model, text, inference_time)

            results[model] = {'success': True, 'time': inference_time}

        except Exception as e:
            print(f"\nModel: {model}")
            print("-" * 60)
            print(f"ERROR: {e}")
            print("=" * 60)
            results[model] = {'success': False, 'error': str(e)}

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, result in results.items():
        if result['success']:
            print(f"{model}: Success, {result['time']:.2f}s")
        else:
            print(f"{model}: ERROR - {result['error']}")
    print("="*60)


if __name__ == "__main__":
    main()
