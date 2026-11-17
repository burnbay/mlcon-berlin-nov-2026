"""The oMLX twin of ai_astrology.py: same Maude, same prompts, thinking on and streamed live, but served
by oMLX over its OpenAI-compatible API instead of Ollama.

    python ai_astrology_omlx.py
    python ai_astrology_omlx.py --no-think

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install openai python-dotenv rich`.
"""
import os
import sys
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

# ============================================================================
#  THINKING (REASONING) ON / OFF  <-- change this one line
#
#    THINK = True    the model reasons first: you see its thinking, then the
#                    answer (slower, often better)
#    THINK = False   straight to the answer (much faster)
#
#  Or leave it and switch it off from the command line:
#    python ai_astrology_omlx.py --no-think
# ============================================================================
THINK = True
if "--no-think" in sys.argv:
    THINK = False


def main():
    today = date.today().strftime('%A, %d-%m-%Y')
    LLM = "mlx-community--Qwen3.5-4B-MLX-4bit"  # oMLX names Hugging Face cache models "org--repo"

    print(f"Thinking is {'ON' if THINK else 'OFF'} (change THINK at the top of ai_astrology_omlx.py, or run with --no-think)\n")
    name = input("Enter your name: ")
    star_sign = input("Enter your star sign: ")

    system_prompt = """You are an AI astrology assistant called Maude. Provide a short but interesting, positive and
        optimistic horoscope for tomorrow. Provide the response in Markdown format.
        Remember, the user is looking for a positive and optimistic outlook on their future.
        Use British English, metric and EU date formats where applicable."""

    instruction = f"Please provide a horoscope for {name} whose star sign is {star_sign}. Today's date is {today}."

    # No max_tokens: like num_predict=-1 in the Ollama version, so the thinking can't eat the whole budget
    stream = client.chat.completions.create(
        model=LLM, stream=True,   # stream=True: tokens arrive as they are generated
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': instruction}],
        temperature=0.8, top_p=0.95,
        extra_body={"top_k": 40,
                    "chat_template_kwargs": {"enable_thinking": THINK}},  # ← thinking on/off: see THINK at the top
    )
    show(stream, "🤔 Maude's Thinking Process:", "✨ Your Horoscope:")


def show(stream, thinking_title, answer_title):
    """Print the thinking live (grey), then the answer live (rendered as Markdown)."""
    console = Console()
    thinking, answer, live = False, "", None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)   # oMLX's name for Ollama's thinking
        if reasoning:                               # reasoning tokens come first
            if not thinking:
                console.print(f"[bold blue]{thinking_title}[/bold blue]")
                thinking = True
            console.print(reasoning, style="dim", end="", markup=False, highlight=False)
        if delta.content:                           # then the answer itself
            if live is None:
                if thinking:
                    console.print("\n\n" + "=" * 50 + "\n")
                console.print(f"[bold magenta]{answer_title}[/bold magenta]")
                live = Live(Markdown(""), console=console, refresh_per_second=12, vertical_overflow="visible")
                live.start()
            answer += delta.content
            live.update(Markdown(answer))
    if live:
        live.stop()

if __name__ == "__main__":
    main()
