"""The oMLX twin of both visual_ollama.py and visual_ml_studio.py (the same demo on two backends): base64-encode
an image and POST it with a question as raw HTTP, thinking off, but to oMLX's OpenAI-compatible endpoint instead
of Ollama's /api/chat or LM Studio's :1234. It asks both originals' questions.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install requests python-dotenv`.
"""
import requests
import json
import base64
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")

HERE = Path(__file__).parent


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None


def send_request(model, prompt, image_path=None):
    url = "http://127.0.0.1:8000/v1/chat/completions"

    user_content = [{"type": "text", "text": prompt}]

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image_ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }

        mime_type = mime_types.get(image_ext, 'image/jpeg')

        base64_image = encode_image_to_base64(image_path)
        if base64_image is None:
            raise Exception(f"Failed to encode image: {image_path}")

        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful visual assistant, give detailed and accurate answers."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}   # "think": False in the Ollama version
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")


def main():
    data_dir = HERE / "data"
    images = [str(data_dir / f"IMG_{i}.jpg") for i in range(1, 13)]

    models = [
        "mlx-community--Qwen3.5-4B-MLX-4bit",   # oMLX names Hugging Face cache models "org--repo"
    ]

    questions = [
        {"prompt": "Using the graph, what is the Read noise for gain=50?",   # the visual_ollama.py question
         "image": images[0],
         },
        {"prompt": "Please output the equation for Markdown",   # the visual_ml_studio.py question
         "image": images[11],
         },
    ]

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 60}")

        for i, question in enumerate(questions):
            prompt = question["prompt"]
            image_path = question["image"]

            print(f"\nQuestion {i + 1}: {prompt}")
            print(f"Image: {os.path.basename(image_path)}")
            print("-" * 40)

            try:
                response = send_request(model_name, prompt, image_path)

                if 'choices' in response and len(response['choices']) > 0:
                    message = response['choices'][0]['message']
                    content = message.get('content')
                    if content:
                        print(f"Response: {content}")
                    elif message.get('reasoning_content'):
                        print("Response: (empty - model spent all tokens on thinking, consider increasing token limit)")
                    else:
                        print("Response: (empty)")
                else:
                    print("Error: Unexpected response format")
                    print(f"Raw response: {response}")

            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
