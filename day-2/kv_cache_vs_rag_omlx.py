"""The oMLX twin of kv_cache_vs_rag.py: the same KV-cache vs RAG lab on Alice, timing the first token, but
the chat model and the all-MiniLM embeddings are both served by oMLX (with its prefix cache) instead of
the class Spark and Ollama.

    python kv_cache_vs_rag_omlx.py                 (chapters I-III, ~8k tokens)
    python kv_cache_vs_rag_omlx.py --whole-book    (~38k tokens, ~35s to read it the first time)

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit and mlx-community/all-MiniLM-L6-v2-bf16
in the Hugging Face cache, and OMLX_API_KEY in the environment or the repo-root .env.
Needs `pip install numpy openai python-dotenv`.
"""
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

HERE = Path(__file__).parent
TEXT = (HERE / "data" / "alice_in_wonderland.txt").read_text()   # ~38k tokens for Qwen3.5
WHOLE_BOOK = "--whole-book" in sys.argv   # the default is chapters I-III (~8k tokens): quick even on a laptop

# oMLX names Hugging Face cache models "org--repo"; Qwen3.8-27B-oQ4e-mtp is sharper but much slower
MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"
EMBED_MODEL = "mlx-community--all-MiniLM-L6-v2-bf16"
if WHOLE_BOOK:
    BOOK = TEXT
    QUESTIONS = [
        "What is written on the little bottle Alice finds?",
        "Who is at the mad tea-party?",
        "What does the Queen use as croquet mallets and balls?",
        "How does the story end?",
    ]
else:
    start = TEXT.index("CHAPTER I.", TEXT.index("CHAPTER I.") + 1)      # skip the table of contents
    BOOK = TEXT[start:TEXT.index("CHAPTER IV.", start)]
    QUESTIONS = [
        "What is written on the little bottle Alice finds?",
        "What is the pool Alice swims in made of?",
        "Who wins the Caucus-race, and what are the prizes?",
        "How many times does Alice change size, and what causes each change?",   # needs the whole text
    ]

# oMLX keeps its prefix cache between runs (in memory and on SSD), and anyone else using the same server
# shares it. Without this, a second run of the script would find the text already cached. A unique tag at
# the very START makes this run's copy of the text different from every earlier one, so your first
# question really is uncached.
MY_TAG = f"[reader {uuid.uuid4().hex[:8]}]\n"


def ask(context, question):
    """Stream the answer and time the FIRST token: that's the prefill, where the cache makes the difference."""
    start, first, answer = time.time(), None, ""
    stream = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=120, stream=True,
        messages=[{"role": "system", "content": "Answer in one or two sentences, using only the text provided."},
                  {"role": "user", "content": f"{context}\n\nQuestion: {question}"}],   # stable text first, question last
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.time()
            answer += chunk.choices[0].delta.content
    return answer.strip(), first - start


def show(label, context, questions):
    print(f"\n=== {label}")
    for q in questions:
        answer, ttft = ask(context, q)
        print(f"first token {ttft:6.2f}s  {q}\n                    -> {answer}")


def embed(texts):
    """all-MiniLM vectors from oMLX, 64 texts per request."""
    return np.array([item.embedding for i in range(0, len(texts), 64)
                     for item in client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 64]).data])


# 1. KV-cache ("CAG"): the whole text every time. The first question pays to read it (prefill, seconds);
#    after that the server reuses the cached keys and values for the identical beginning (a fraction of a second).
show("Whole text: first question uncached, then from the KV-cache", MY_TAG + BOOK, QUESTIONS)

# 2. The cache is a PREFIX cache: change anything at the start and everything after it must be recomputed.
now = f"Today is {datetime.now():%A %d %B %Y, %H:%M:%S}.\n"
show("Change the START (a timestamp before the text): cache miss", now + MY_TAG + BOOK, QUESTIONS[:1])
show("Change the END (the same timestamp after the text): cache hit", MY_TAG + BOOK + "\n" + now, QUESTIONS[:1])

# 3. RAG: chunk, embed, and send only the 4 chunks closest to the question (small prompt, but it can miss things)
chunks = [BOOK[i:i + 1500] for i in range(0, len(BOOK), 1200)]            # 1,500 characters, 300 overlap
vectors = embed(chunks)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
print("\n=== RAG: only the 4 best-matching chunks")
for q in QUESTIONS:
    qv = embed([q])[0]
    best = np.argsort(vectors @ (qv / np.linalg.norm(qv)))[-4:]
    answer, ttft = ask("\n...\n".join(chunks[i] for i in sorted(best)), q)
    print(f"first token {ttft:6.2f}s  {q}\n                    -> {answer}")
