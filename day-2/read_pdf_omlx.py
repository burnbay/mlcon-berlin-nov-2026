"""The oMLX twin of read_pdf.py: same page, same prompts, thinking still off, but the vision model is
served by oMLX over its OpenAI-compatible API instead of Ollama.

    python read_pdf_omlx.py          (page 8, German edition)
    python read_pdf_omlx.py 9        (page 9)
    python read_pdf_omlx.py 8 en     (page 8, English edition)

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install pymupdf openai python-dotenv`.
"""
import base64
import os
import re
import sys
from pathlib import Path

import pymupdf       # PyMuPDF: turns a PDF page into an image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

HERE = Path(__file__).parent
# The Berlin Senate publishes the same report in German and English, page for page.
# The German one is the default: the model reads German and you ask in English (or any language).
EDITIONS = {"de": "land-berlin-2026.pdf",      # "Land Berlin – Investorenpräsentation, August 2026"
            "en": "state-of-berlin-2026.pdf"}  # "State of Berlin – Investor Presentation, August 2026"
FRONT_MATTER = 0   # printed page numbers match the file (a report with front matter numbered i-iv would need 4 here)
PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 8   # page 8: a line chart of real GDP growth, Berlin vs Germany, 2007-2025
EDITION = sys.argv[2] if len(sys.argv) > 2 else "de"   # python read_pdf_omlx.py 8 en  -> the English edition
PDF = HERE / "data" / EDITIONS[EDITION]
MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"   # oMLX names Hugging Face cache models "org--repo"
TRANSCRIBE = ("Transcribe this page as Markdown. Keep headings, paragraphs and lists. Turn any table into a "
              "Markdown table, and describe any chart or picture in one short paragraph of text (never an image link). "
              "Output only the Markdown.")

# 1. Render just one page: reading all 33 pages would take minutes
page = pymupdf.open(PDF)[PAGE - 1 + FRONT_MATTER]   # printed page number -> position in the file
png = page.get_pixmap(dpi=150).tobytes("png")
image_url = "data:image/png;base64," + base64.b64encode(png).decode()   # OpenAI-style APIs take images as data URLs


def ask(prompt):
    reply = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt},
                                               {"type": "image_url", "image_url": {"url": image_url}}]}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (reply.choices[0].message.content or "").strip().removeprefix("```markdown").removesuffix("```").strip()


# 2. The page as Markdown, saved next to this script (with -omlx so it doesn't overwrite the Ollama run)
markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)\n*", "", ask(TRANSCRIBE))   # small models like to invent image links
out = HERE / f"berlin-{EDITION}-page-{PAGE}-omlx.md"
out.write_text(markdown + "\n")
print(markdown, f"\n\n(saved to {out.name})\n")

# 3. Ask questions about the chart or pictures on the page (Enter on its own to finish)
while question := input("Ask about this page (Enter to finish): ").strip():
    print(ask(f"Look at the chart or picture on this page. {question} Answer briefly."), "\n")
