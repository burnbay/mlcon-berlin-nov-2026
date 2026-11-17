"""The oMLX twin of mlx_quick_test.py: the same eight quick questions, in six languages, with greedy
decoding (same answers every run), but served by oMLX instead of loading the weights with mlx_lm.

oMLX has no LFM2.5-1.2B, so this uses the nearest model it does serve: Qwen3.5-2B at 8-bit (thinking off).

Needs oMLX running on :8000 with mlx-community/Qwen3.5-2B-8bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env.

    pip install openai python-dotenv
    python day-1/mlx_quick_test_omlx.py
"""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

MODEL = "mlx-community--Qwen3.5-2B-8bit"
SYSTEM = "Answer in English only. Give just the short, final answer with no explanation."


def ask(question):
    start, answer = time.time(), ""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        max_tokens=128,
        temperature=0.0,  # deterministic: same answers every run
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content
    return answer.strip().split("\n")[0], (time.time() - start) * 1000  # first line only


questions = [
    "Which British king abdicated in 1936?",
    "Wann war das erste Oktoberfest?",
    "¿Dónde nació Salvador Dalí?",
    "Lequel des deux est le plus ancien, la Tour Eiffel ou le Sacré-Cœur ?",
    "什么是爱因斯坦最著名的方程式？",
    "ما هي عاصمة مصر؟",
    "Write one line of Java code that prints Hello World to the console.",
    "What is the derivative of 6x^2-5x+12?",
]

if __name__ == "__main__":
    ask("Hi")  # warm up so the first timing is fair
    for q in questions:
        answer, ms = ask(q)
        print(f"{q}\n  -> {answer}  ({ms:.0f} ms)\n")
