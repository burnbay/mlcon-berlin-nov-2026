"""The oMLX twin of grimm_fairy_tales_rag_demo.py: the same Qwen3-Embedding search (batched, 512-dim, top-k)
and the same Qwen3.5 answers, but both are served by oMLX over its OpenAI-compatible API instead of
transformers + Ollama, so there's no torch to install.

Needs oMLX running on :8000 with mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ and
mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and OMLX_API_KEY in the environment or the
repo-root .env. Needs `pip install openai python-dotenv numpy rich tiktoken`.
"""
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
import tiktoken

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

EMBEDDING_MODEL = 'mlx-community--Qwen3-Embedding-0.6B-4bit-DWQ'   # oMLX names Hugging Face cache models "org--repo"
EMBEDDING_DIM = 512
BATCH_SIZE = 64
MAX_TOKEN_LENGTH = 512
TOP_K = 6
LLM = "mlx-community--Qwen3.5-4B-MLX-4bit"
HERE = Path(__file__).parent


def get_embeddings(texts: List[str]) -> np.ndarray:
    # oMLX does the Qwen3-Embedding recipe server-side: it appends <|endoftext|>, pools the last token,
    # truncates each text to max_length tokens and cuts the vector to `dimensions` (Matryoshka-style)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIM,
        extra_body={"max_length": MAX_TOKEN_LENGTH}
    )

    embeddings = np.array([item.embedding for item in response.data])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def chunk_text(text: str) -> List[Dict[str, Any]]:
    # Grimm text uses quadruple newlines as story separators
    chapters = re.split(r'\n\n\n\n', text)

    encoder = tiktoken.get_encoding("cl100k_base")

    chunks = []
    for i, chapter in enumerate(chapters):
        cleaned = ' '.join(chapter.strip().split())
        if len(cleaned) > 100:
            token_count = len(encoder.encode(cleaned))
            chunks.append({
                'id': i,
                'text': cleaned,
                'length': len(cleaned),
                'tokens': token_count
            })

    return chunks


def search_chunks(query: str, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray) -> List[Dict[str, Any]]:
    query_embeddings = get_embeddings([query])
    query_embedding = query_embeddings[0]

    similarities = chunk_embeddings @ query_embedding

    top_indices = np.argsort(similarities)[::-1][:min(TOP_K, len(similarities))]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        chunk = chunks[idx]
        results.append({
            'text': chunk['text'],
            'similarity': score,
            'length': chunk['length'],
            'tokens': chunk['tokens']
        })

    return results


def generate_all_embeddings(chunks: List[Dict[str, Any]]) -> np.ndarray:
    print(f"Generating embeddings for {len(chunks)} chunks...")

    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_texts = [chunk['text'] for chunk in batch_chunks]
        batch_embeddings = get_embeddings(batch_texts)
        all_embeddings.append(batch_embeddings)

    return np.concatenate(all_embeddings, axis=0)


def display_results(query: str, results: List[Dict[str, Any]]) -> None:
    print(f"\nQuery: '{query}'")
    print("=" * 50)

    total_tokens = 0
    for i, result in enumerate(results, 1):
        similarity = result['similarity']
        text = result['text']
        tokens = result['tokens']
        total_tokens += tokens
        display_text = text[:100] + "..." if len(text) > 100 else text
        print(f"{i}. [Score: {similarity:.3f}] [Tokens: {tokens}] {display_text}")

    print(f"\nTotal tokens in results: {total_tokens}")
    print()


def generate_answer(query: str, results: List[Dict[str, Any]]) -> None:
    console = Console()

    # Only use top 3 results to keep context focused
    context = "\n\n".join([f"Context {i+1}: {result['text'][:1000]}..." if len(result['text']) > 1000
                           else f"Context {i+1}: {result['text']}"
                           for i, result in enumerate(results[:3])])

    system_prompt = """You are a helpful assistant that answers questions about Grimm fairy tales.
    Use the provided context from the fairy tales to answer the user's question accurately and concisely.
    Provide your answer in English, even though the source text may be in German.
    Keep your response short and focused on directly answering the question."""

    instruction = f"""Based on the following context from Grimm fairy tales, please answer this question: {query}

Context:
{context}

Please provide a short, direct answer in English."""

    try:
        response = client.chat.completions.create(
            model=LLM,
            stream=False,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': instruction}
            ],
            temperature=0.7,
            top_p=0.95,
            extra_body={"top_k": 40, "chat_template_kwargs": {"enable_thinking": False}}
        )
        message = response.choices[0].message

        console.print("\n[bold green]RAG Answer:[/bold green]")

        thinking = getattr(message, 'reasoning_content', None)
        if thinking:
            console.print(f"[dim]Thinking: {thinking[:200]}...[/dim]")

        console.print(f"[blue]{message.content}[/blue]")

    except Exception as e:
        console.print(f"[red]Error generating answer: {e}[/red]")


def main() -> None:
    print("Simple Grimm Tales RAG Search Demo")
    print("=" * 40)

    print(f"Embedding with {EMBEDDING_MODEL} on oMLX")

    try:
        with open(HERE / "data" / "Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt", "r", encoding="utf8") as f:
            text = f.read()
        print(f"Loaded document: {len(text):,} characters")
    except FileNotFoundError:
        print("Error: 'data/Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt' not found!")
        return

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    start_time = time.time()
    chunk_embeddings = generate_all_embeddings(chunks)
    embed_time = time.time() - start_time
    print(f"Embeddings generated in {embed_time:.1f}s")

    queries = [
        "What did the frog king promise the princess in exchange for her golden ball?",
        "What happened to Hansel and Gretel in the forest?",
        "What did Little Red Riding Hood's mother tell her to do?"
    ]

    print("\n" + "=" * 60)
    print("SEARCH DEMO")
    print("=" * 60)

    for query in queries:
        start_time = time.time()
        results = search_chunks(query, chunks, chunk_embeddings)
        search_time = time.time() - start_time

        display_results(query, results)
        print(f"Search completed in {search_time*1000:.0f}ms")

        generate_answer(query, results)
        print("-" * 50)

    print("Demo completed!")


if __name__ == "__main__":
    main()
