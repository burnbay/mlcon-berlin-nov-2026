"""The oMLX twin of analyse_sentiment_02.py: the same three texts and the same post-processing, thinking
still off, but the raw HTTP call goes to oMLX's OpenAI-compatible endpoint instead of Ollama's
/api/generate.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install requests python-dotenv`.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")


def analyse_sentiment(text, model="mlx-community--Qwen3.5-4B-MLX-4bit"):  # oMLX names models "org--repo"
    prompt = f"""
    Analyse the sentiment of the following text and respond with exactly one word:
    'positive', 'neutral', or 'negative'.
    Text: {text}
    Sentiment:
    """

    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    sentiment = (result["choices"][0]["message"]["content"] or "").strip().lower()

    if sentiment not in ["positive", "neutral", "negative"]:
        if "positive" in sentiment:
            sentiment = "positive"
        elif "negative" in sentiment:
            sentiment = "negative"
        else:
            sentiment = "neutral"
    return sentiment


if __name__ == "__main__":
    texts = [
        "I had a wonderful day today!",
        "The weather is cloudy.",
        "This is the worst service I've ever experienced."
    ]

    for text in texts:
        sentiment = analyse_sentiment(text)
        print(f"Text: '{text}'")
        print(f"Sentiment: {sentiment}")
        print("-" * 50)
