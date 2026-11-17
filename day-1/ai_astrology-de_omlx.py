"""The oMLX twin of ai_astrology-de.py: das lustige deutsche Horoskop, same Oma Gerda, same prompts,
thinking on and streamed live, but served by oMLX over its OpenAI-compatible API instead of Ollama.

    python ai_astrology-de_omlx.py
    python ai_astrology-de_omlx.py --no-think

Same app as ai_astrology_omlx.py (Qwen3.5-4B via oMLX, thinking on, rendered with rich);
only the prompts change. Try changing Oma Gerda's personality in SYSTEM_PROMPT, or
turn the temperature up for wilder predictions.

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

LLM = "mlx-community--Qwen3.5-4B-MLX-4bit"  # oMLX names Hugging Face cache models "org--repo"

# ============================================================================
#  THINKING (REASONING) ON / OFF  <-- change this one line
#
#    THINK = True    the model reasons first: you see its thinking, then the
#                    answer (slower, often better)
#    THINK = False   straight to the answer (much faster)
#
#  Or leave it and switch it off from the command line:
#    python ai_astrology-de_omlx.py --no-think
# ============================================================================
THINK = True
if "--no-think" in sys.argv:
    THINK = False
WOCHENTAGE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]

SYSTEM_PROMPT = """Du bist Oma Gerda, eine KI-Astrologin aus Berlin-Neukölln mit echter Berliner Schnauze.
Du schreibst ein kurzes, lustiges und trotzdem positives Horoskop für morgen, auf Deutsch.

Dein Stil:
- Herzlich, aber frech: ein bisschen Berliner Dialekt ("wa?", "Icke", "Keene Panik", "dit"),
  aber so, dass alle es verstehen.
- Die Sterne sagen Alltägliches mit großem kosmischem Ernst voraus: die S-Bahn kommt pünktlich
  (ein Wunder!), ein Parkplatz in Kreuzberg, die perfekte Currywurst, ein Termin beim Bürgeramt,
  ein Pfandbon, der Nachbar, der endlich die Musik leiser macht.
- Gern ein Seitenhieb auf den Flughafen BER oder auf Excel-Tabellen, nie gemein gegen Menschen.
- Am Ende eine Markdown-Liste mit drei Punkten: **Glückszahl**, **Glücksfarbe** und **Kosmischer Tipp**.
- Maximal 150 Wörter. Antworte in Markdown, mit einer Überschrift.
- Immer optimistisch: der Mensch soll lachen und sich auf morgen freuen."""


def main():
    heute = date.today()
    datum = f"{WOCHENTAGE[heute.weekday()]}, {heute.strftime('%d.%m.%Y')}"

    print(f"Thinking is {'ON' if THINK else 'OFF'} (change THINK at the top of ai_astrology-de_omlx.py, or run with --no-think)\n")
    name = input("Wie heißt du? ")
    sternzeichen = input("Dein Sternzeichen? ")

    anweisung = (f"Bitte schreib ein Horoskop für {name}, Sternzeichen {sternzeichen}. "
                 f"Heute ist {datum}.")

    # No max_tokens: like num_predict=-1 in the Ollama version, so the thinking can't eat the whole budget
    stream = client.chat.completions.create(
        model=LLM, stream=True,   # stream=True: tokens arrive as they are generated
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": anweisung}],
        temperature=0.9, top_p=0.95,
        extra_body={"top_k": 40,
                    "chat_template_kwargs": {"enable_thinking": THINK}},  # ← thinking on/off: see THINK at the top
    )
    show(stream, "🤔 Oma Gerdas Gedanken:", "🔮 Dein Horoskop:")


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
