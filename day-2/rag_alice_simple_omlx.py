"""The oMLX twin of rag_alice_simple.py: the same minimal RAG loop on Alice in Wonderland (same chunks,
prompts and questions, thinking still off), but the embeddings and the answers come from oMLX's
OpenAI-compatible API instead of Ollama. Still raw `requests`, so you can see every HTTP call.

Needs oMLX running on :8000 with mlx-community/embeddinggemma-300m-bf16 and
mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and OMLX_API_KEY in the environment or the
repo-root .env. Needs `pip install numpy requests python-dotenv`.
"""
import os
import re
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
HEADERS = {"Authorization": f"Bearer {api_key}"}

EMBEDDING_MODEL = "mlx-community--embeddinggemma-300m-bf16"   # oMLX names Hugging Face cache models "org--repo"
LLM_MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"
OMLX_URL = "http://127.0.0.1:8000/v1"
BATCH_SIZE = 32   # chunks per /embeddings request: one call per batch instead of one per chunk
HERE = Path(__file__).parent


def get_embeddings(texts, task_type="document"):
    # EmbeddingGemma uses task-specific prompts for document vs query
    if task_type == "query":
        prompts = [f"task: search result | query: {text}" for text in texts]
    else:
        prompts = [f"title: none | text: {text}" for text in texts]

    response = requests.post(
        f"{OMLX_URL}/embeddings",
        headers=HEADERS,
        json={"model": EMBEDDING_MODEL, "input": prompts}
    )
    response.raise_for_status()
    return [np.array(item["embedding"]) for item in response.json()["data"]]


def get_embedding(text, task_type="document"):
    return get_embeddings([text], task_type)[0]


def chunk_text(text, chunk_size=250, overlap=50):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = len(sentence.split())

        if word_count + words > chunk_size and word_count > 0:
            chunks.append(" ".join(current_chunk))
            overlap_text = " ".join(current_chunk)
            overlap_words = overlap_text.split()[-overlap:]
            current_chunk = [" ".join(overlap_words)]
            word_count = len(overlap_words)

        current_chunk.append(sentence)
        word_count += words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def query_rag(question, chunks, embeddings, top_k=6):
    query_emb = get_embedding(question, "query")

    scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[-top_k:][::-1]

    print(f"Q: {question}\n")
    print("Top matches:")
    for i, idx in enumerate(top_indices, 1):
        preview = chunks[idx][:80].replace('\n', ' ') + "..."
        print(f"  {i}. Chunk {idx:03d} | Score: {scores[idx]:.4f} | \"{preview}\"")
    print()

    context = "\n\n".join([f"[{i}] {chunks[idx]}" for i, idx in enumerate(top_indices, 1)])

    prompt = f"""Answer the question using the context below. Be specific and quote relevant text.

Question: {question}

Context:
{context}

Answer:"""

    response = requests.post(
        f"{OMLX_URL}/chat/completions",
        headers=HEADERS,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.05,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]
    print(f"A: {answer}\n")
    print("-" * 70)
    return answer


def main():
    print("Simple RAG Demo - Alice in Wonderland\n")

    with open(HERE / "data" / "alice_in_wonderland.txt", 'r', encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r'\r\n', '\n', text)
    match = re.search(r'CHAPTER I\.\s*\n\s*[\w\s-]+\n\s*\n', text)
    if match:
        text = text[match.end():]

    print("Creating chunks...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks\n")

    print("Generating embeddings...")
    embeddings = []
    for start in range(0, len(chunks), BATCH_SIZE):
        embeddings.extend(get_embeddings(chunks[start:start + BATCH_SIZE], "document"))
        print(f"  {len(embeddings)}/{len(chunks)} embedded")
    print(f"Done! {len(chunks)} chunks ready.\n")
    print("-" * 70)

    questions = [
        "What was Alice doing at the beginning of the story?",
        "What was written on the bottle that made Alice shrink?",
        "Where was the Cheshire Cat when Alice first met him?",
        "What did the White Rabbit say when Alice first saw him?",
    ]

    for question in questions:
        query_rag(question, chunks, embeddings)


if __name__ == "__main__":
    main()
