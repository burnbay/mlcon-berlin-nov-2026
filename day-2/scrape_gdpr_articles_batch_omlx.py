#!/usr/bin/env python3
"""The oMLX twin of scrape_gdpr_articles_batch.py: scrape GDPR articles 20-25 and summarise each, but the
summaries come from Qwen3.5-4B served by oMLX over its OpenAI-compatible API instead of Ollama.

    python scrape_gdpr_articles_batch_omlx.py          (reads the saved pages in data/gdpr-info/)
    python scrape_gdpr_articles_batch_omlx.py --live   (fetches them from gdpr-info.eu)

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install requests beautifulsoup4 python-dotenv`.
"""

import os
import requests
from bs4 import BeautifulSoup
import re
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")

HERE = Path(__file__).parent
OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"   # oMLX names Hugging Face cache models "org--repo"


def fetch_page(url, headers):
    """The room's internet is slow, so read the saved copy in data/gdpr-info/ unless --live is given."""
    saved = HERE / "data" / "gdpr-info" / (url.rstrip("/").rsplit("/", 1)[-1] + ".html")
    if saved.exists() and "--live" not in sys.argv:
        print(f"Reading the saved copy of {url} (add --live to fetch it)")
        return saved.read_text(encoding="utf-8")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def scrape_article_text(article_num: int) -> tuple[str, str] | None:
    """Scrape a single GDPR article page. Returns (title, full_text) or None."""
    url = f"https://gdpr-info.eu/art-{article_num}-gdpr/"
    print(f"  Scraping {url} …", end=" ", flush=True)

    try:
        html = fetch_page(url, HEADERS)
    except requests.RequestException as e:
        print(f"FAILED ({e})")
        return None

    soup = BeautifulSoup(html, "html.parser")

    # --- title ---
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else f"Art. {article_num} GDPR"

    # --- full article body ---
    # The main content lives inside <div class="entry-content"> on this site.
    entry = soup.find("div", class_="entry-content")
    if not entry:
        # fallback: grab everything inside <article>
        entry = soup.find("article")
    if not entry:
        print("FAILED (no content container)")
        return None

    # Gather all paragraph / list-item text, skipping nav / footer pollution.
    paragraphs: list[str] = []
    for tag in entry.find_all(["p", "li", "h2", "h3"]):
        # skip children of nav / footer inside the entry (rare but safe)
        if tag.find_parent(["nav", "footer"]):
            continue
        text = tag.get_text(strip=True)
        if text and len(text) > 10:
            paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)
    if not full_text:
        print("FAILED (empty body)")
        return None

    print(f"OK  ({len(full_text):,} chars)")
    return title, full_text


def summarise_with_omlx(title: str, full_text: str) -> str:
    """Ask Qwen3.5-4B for a concise 1-2 sentence summary of the article."""
    prompt = (
        "You are a legal assistant. Below is the full text of a GDPR article.\n\n"
        f"Title: {title}\n\n"
        f"{full_text}\n\n"
        "Write a concise, plain-English summary in ONE or TWO sentences that captures "
        "what this article is about. Return ONLY the summary — no preamble, no markdown, no quotes."
    )

    # Thinking off: only the summary is kept, and a thinking 4B model can take minutes per article
    # (oMLX keeps a slow request alive, so the 120 s timeout below doesn't cut it short)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = requests.post(OMLX_URL, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        summary = (data["choices"][0]["message"]["content"] or "").strip()
        # Clean up common LLM verbosity
        summary = re.sub(r'^["\']|["\']$', '', summary)
        summary = re.sub(r'^(Here( is|’s)|Summary:|The summary)', '', summary, flags=re.IGNORECASE).strip()
        return summary
    except Exception as e:
        print(f"    oMLX error: {e}")
        return "(summary unavailable)"


def main():
    articles = range(20, 26)  # 20–25 inclusive
    results: list[dict] = []

    for num in articles:
        print(f"\nArticle {num}:")
        scraped = scrape_article_text(num)
        if scraped is None:
            print("  Skipping — could not scrape.")
            continue
        title, full_text = scraped

        print(f"  Summarising with {MODEL} …", end=" ", flush=True)
        summary = summarise_with_omlx(title, full_text)
        print(f"OK  ({len(summary)} chars)")

        results.append({
            "article": f"Art. {num}",
            "summary": summary,
            "full_article": full_text,
        })

    # With _omlx so it doesn't overwrite the Ollama run (gdpr_viewer_omlx.py reads this one)
    out_path = HERE / "data" / "gdpr_articles_20_25_omlx.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"GDPR": results}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} articles to {out_path}")


if __name__ == "__main__":
    main()
