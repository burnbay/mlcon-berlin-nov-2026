"""The oMLX twin of embedding_demo.py: the same tea / coffee / mud / dirt similarity grid, but the all-MiniLM
embeddings come from oMLX over its OpenAI-compatible API instead of Ollama.

Needs oMLX running on :8000 with mlx-community/all-MiniLM-L6-v2-bf16 in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv numpy tabulate`.
"""
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)


def get_embeddings(texts):
    # one request embeds every word; oMLX names Hugging Face cache models "org--repo"
    response = client.embeddings.create(model="mlx-community--all-MiniLM-L6-v2-bf16", input=texts)
    return [item.embedding for item in response.data]


def main():
    words = ["tea", "coffee", "mud", "dirt"]

    embeddings = np.array(get_embeddings(words))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(embeddings, embeddings.T)

    header = [""] + words
    table = [
        [words[i]] + [f"{similarities[i][j]:.3f}" for j in range(len(words))]
        for i in range(len(words))
    ]

    print(tabulate(table, headers=header, tablefmt="grid"))


if __name__ == "__main__":
    main()
