"""The oMLX twin of formatted_response_example.py: the numbers 1-10 in nine languages as JSON, thinking
still off, but served by oMLX over its OpenAI-compatible API, with JSON mode as `response_format`.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv`.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)


def generate_formatted_response(prompt):
    try:
        response = client.chat.completions.create(
            model="mlx-community--Qwen3.5-4B-MLX-4bit",  # oMLX names Hugging Face cache models "org--repo"
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},  # Ollama's format="json"
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return None


def main():
    prompt = """List the numbers from 1 to 10 and their names in
    English, French, German, Dutch, Chinese, Russian, Arabic, Polish, Hungarian.
    Provide the output in this exact JSON format:
    {
      "numbers": [
        {
          "number": 7,
          "English": "seven",
          "French": "sept",
          "German": "sieben",
          "Dutch": "zeven",
          "Chinese": "七",
          "Russian": "семь",
          "Arabic": "سبعة",
          "Polish": "siedem",
          "Hungarian": "hét"
        },
        ...and so on for numbers 1-10
      ]
    }"""

    response = generate_formatted_response(prompt)

    try:
        if response:
            parsed = json.loads(response)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("Received non-JSON response:")
        print(response)


if __name__ == "__main__":
    main()
