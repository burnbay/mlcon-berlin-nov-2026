"""The oMLX twin of model_test.py: the same five tricky questions, each with an automatic check, run with
thinking off and on, but against models served by oMLX instead of Ollama.

    python model_test_omlx.py                                   (Qwen3.5 0.8B and 4B)
    python model_test_omlx.py mlx-community--Qwen3.5-2B-4bit    (any ids from /v1/models)
    python model_test_omlx.py --all                             (every chat model oMLX serves)
    python model_test_omlx.py --all --big                       (... including the big and partial ones)

Needs oMLX running on :8000 with mlx-community/Qwen3.5-0.8B-4bit and mlx-community/Qwen3.5-4B-MLX-4bit in the
Hugging Face cache, and OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv`.
"""
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")

# Test models on YOUR questions: every question has a check, so the score is automatic.
# Works with anything OpenAI-compatible; here it's oMLX. Add or change models freely.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key, timeout=600)
MODELS = [a for a in sys.argv[1:] if not a.startswith("--")] or ["mlx-community--Qwen3.5-0.8B-4bit",
                                                                 "mlx-community--Qwen3.5-4B-MLX-4bit"]
if "--all" in sys.argv:
    # Chat models only (oMLX tags each model llm, vlm, embedding, ...). The 27B is ~18 GB and slow, and
    # gemma-4-31b is only partly downloaded here, so both need --big as well.
    BIG = {"Qwen3.8-27B-oQ4e-mtp", "mlx-community--gemma-4-31b-it-4bit"}
    MODELS = [m.id for m in client.models.list()
              if getattr(m, "model_type", None) in ("llm", "vlm") and ("--big" in sys.argv or m.id not in BIG)]

QUESTIONS = [
    ("What is 37*27*1001? Reply with just the number.", lambda a: "999999" in a.replace(",", "")),
    ("Is 99,409 a prime number? Answer yes or no.", lambda a: re.search(r"\byes\b", a, re.I) is not None),
    ("How many letter r's are in the word 'strawberry'? Reply with just the number.", lambda a: re.search(r"\b3\b|three", a, re.I) is not None),
    ("Which is heavier: a kilo of feathers or a kilo of bricks?", lambda a: re.search(r"same|neither|equal|both", a, re.I) is not None),
    ("I need to wash my car. The car wash is 50 meters away. Should I walk or drive? Answer in one word.",
     lambda a: re.search(r"\bdrive\b", a, re.I) is not None),
]

# Thinking off vs on: the chat template's enable_thinking switch (Ollama's reasoning_effort "none" / "medium")
THINKING = [False, True]

print(f"{'model':36} {'thinking':9} {'score':6} {'ran out':8} {'time':>6}")
for model in MODELS:
    for think in THINKING:
        score, ran_out, start = 0, 0, time.time()
        for question, check in QUESTIONS:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": question}],
                max_tokens=4096,                # room for the reasoning, not just the answer
                extra_body={"chat_template_kwargs": {"enable_thinking": think}},
            )
            score += check(r.choices[0].message.content or "")
            ran_out += r.choices[0].finish_reason == "length"   # still thinking when the budget ran out
        print(f"{model:36} {'on' if think else 'off':9} {score}/{len(QUESTIONS):<4} {ran_out:<8} {time.time() - start:5.0f}s")
