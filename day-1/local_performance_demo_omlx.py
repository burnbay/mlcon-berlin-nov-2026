"""The oMLX twin of local_performance_demo.py: the same multilingual "difficult" questions through
Qwen3.5-4B (thinking off, temperature 0), timed per question, but served by oMLX instead of Ollama.
oMLX also reports its own timings in the usage block: time to first token, and prompt and generation
speed in tokens/second.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv`.
"""
import os, time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

def time_execution(func):
    def wrapper(i, task):
        start = time.time()
        result = func(task)
        elapsed = (time.time() - start) * 1000
        usage = result.usage
        print(f"Task {i+1}: {task}:\n\t\t{result.choices[0].message.content} ({elapsed:.2f} ms; "
              f"first token {getattr(usage, 'time_to_first_token', 0) * 1000:.0f} ms, "
              f"prompt {getattr(usage, 'prompt_tokens_per_second', 0):.0f} tok/s, "
              f"generation {getattr(usage, 'generation_tokens_per_second', 0):.1f} tok/s)\n")
        return result
    return wrapper

run_query = time_execution(lambda p: client.chat.completions.create(model="mlx-community--Qwen3.5-4B-MLX-4bit", temperature=0,
                          messages=[{"role": "system", "content": "Always reply in English, regardless of the question's language. Give only the final answer, no explanation,no translation, no restating the question."},
                                    {"role": "user", "content": p}],
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}}))

run_query(-1, "") # Initialise model so as not to screw up the first timing

tasks = [
    "Which British king abdicated in the 1936",
    "Wann war das erste Oktoberfest?",
    "¿Dónde nació Salvador Dalí?",
    "Lequel des deux est le plus ancien, la Tour Eiffel ou le Sacré-Cœur ?",
    "什么是爱因斯坦最著名的方程式？",
    "في أي بلد يقع كراكاتوا؟",
    "Write a single line in Java to print 'Hello World'",
    "Write a single line command for Linux to count the number for 20-letter words in the dictionary",
    "What is the derivative of 6x^2-5x+12?"
]

for i, task in enumerate(tasks):
    run_query(i, task)
