"""The oMLX twin of analyse_sentiment_01.py: the same one-word sentiment loop with the same light string
cleanup, thinking still off, but the raw HTTP call goes to oMLX's OpenAI-compatible endpoint instead of
Ollama's /api/generate.

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

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        sentiment = (result["choices"][0]["message"]["content"] or "").strip().lower()

        for word in ['positive', 'negative', 'neutral']:
            if word in sentiment:
                return word

        return sentiment if sentiment else "unknown"

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to oMLX: {e}")
        return "error"
    except Exception as e:
        print(f"Error processing response: {e}")
        return "error"


def main():
    print("Sentiment Analysis Tool")
    print("=======================")
    print("This tool uses a local model served by oMLX to analyze text sentiment.")
    print("Make sure oMLX is running...\n")

    while True:
        user_text = input("Enter text to analyze (or 'quit' to exit): ").strip()

        if user_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_text:
            print("Please enter some text to analyze.\n")
            continue

        print("\nAnalyzing sentiment...")
        sentiment = analyse_sentiment(user_text)

        if sentiment == "error":
            print("Failed to analyze sentiment. Please check that oMLX is running.\n")
        else:
            print(f"Sentiment: {sentiment.upper()}\n")


if __name__ == "__main__":
    main()
