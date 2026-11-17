# Day 2 — Applications: RAG, Extraction, Tools, Vibe Coding & Agents

The Day‑2 code from the **MLCon Berlin 2026 "Hands‑On GenAI Development Bootcamp"**.
These scripts back the Day‑2 slide deck (`MLCon Berlin 2026 - Hands-on GenAI
Development Bootcamp Day 2`) and follow its narrative: now that you know what an
LLM *is* (Day 1), what can you *build* with one? The day moves from **RAG**
(retrieval‑augmented generation) through **summarisation, data extraction and
sentiment**, **structured (JSON) output**, **code generation → vibe coding**,
**tool / function calling**, the **Model Context Protocol (MCP)**, and finishes on
modern **agents**.

The recurring theme: almost everything here can be done **locally, privately and
cheaply** with small open‑weight models — you don't need a frontier API.

Every file is a **standalone program** — there is no build step and no package to
install. Run each one directly with `python <script>.py`; it pulls in only the
libraries that one demo needs. [`requirements.txt`](requirements.txt) in this
folder is a convenience superset that installs everything Day 2 needs at once
(`pip install -r day-2/requirements.txt`); you don't need all of it for any single
demo.

## Prerequisites

Most demos talk to a model. Depending on the script you need one (or more) of:

| Backend | How to get it | Used by |
|---|---|---|
| **Ollama** (`127.0.0.1:11434`) | Install from <https://ollama.com>, then `ollama pull qwen3.5:4b`, plus an embedder: `ollama pull embeddinggemma` (and/or `all-minilm`) | most local demos |
| **LM Studio** (`127.0.0.1:1234`, OpenAI‑compatible) | Install, download a vision model (`qwen3.5-4b`), start the local server | `find_beer.py`, `visual_ml_studio.py`, `test_lmstudio_vision.py` |
| **MLX** | Apple‑Silicon only; weights load straight from disk | `chatterbox_tts.py` (TTS) |
| **oMLX** (`127.0.0.1:8000`, OpenAI‑compatible) | Apple‑Silicon only: install from <https://github.com/jundot/omlx>, set an API key, put it in the repo‑root `.env` as `OMLX_API_KEY`, then download the models (below) | every `*_omlx.py` twin, `omlx_function_support.py`, `test_omlx_vision.py` |
| **Cloud APIs** | An API key per provider (see below) | the `*_groq.py` twins |
| **Kaggle** | `pip install kagglehub` — datasets download on first run | `payroll.py`, `*_kaggle.py`, `kaggle_summary_complete.py` |

**No Ollama or LM Studio?** Every demo that uses them (or MLX) has an **oMLX twin** next to it, named
`*_omlx.py`. oMLX serves models straight from the Hugging Face cache, named `org--repo`; after downloading,
use "Reload models" on its admin page (or restart it). The Day‑2 twins use these (~11 GB in total):

```
hf download mlx-community/Qwen3.5-4B-MLX-4bit             # the workhorse: chat, vision and tools
hf download mlx-community/Qwen3.5-2B-4bit                 # find_beer_omlx.py
hf download mlx-community/Qwen3.5-2B-8bit                 # find_beer_omlx.py
hf download mlx-community/embeddinggemma-300m-bf16        # RAG embeddings
hf download mlx-community/all-MiniLM-L6-v2-bf16           # kv_cache_vs_rag_omlx.py
hf download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ   # grimm_fairy_tales_rag_demo_omlx.py
hf download mlx-community/chatterbox-turbo-fp16           # text to speech and voice cloning (not 8-bit: it can't clone)
hf download mlx-community/whisper-large-v3-turbo-asr-4bit # transcribes the voice samples for cloning
```

**Recommended models** — `qwen3.5:4b` is the workhorse LLM; for embeddings use
`embeddinggemma` or `all-minilm` (some RAG demos pull a `Qwen/Qwen3-Embedding`
model from Hugging Face instead). Newer is almost always better; anything older
than ~3 months is "too old".

**Cloud keys** — the `*_groq.py` scripts call `load_dotenv()`, so put your key in
a `.env` file in the **repo root** (copy `../.env.example` to `../.env`). Run the
scripts from the repo root — e.g. `python day-2/data_extraction_groq.py` — so the
`.env` is picked up:

```
GROQ_API_KEY=...
```

Sample books, images, audio and datasets live in **`data/`**.

---

## 1. Recap — multimodal: image → text

We open by revisiting Day 1's vision work and pushing it further: a vision model
can read charts, diagrams, sheet music, menus and photos.

| Script | What it shows |
|---|---|
| `visual_ollama.py` | The core **image → text** demo: base64‑encode an image and POST it to Ollama's vision model, asking it to describe the chart (the IEEE "Responsible AI risks" bar chart from the slides). |
| `visual_ml_studio.py` | The same idea against **LM Studio**'s OpenAI‑compatible endpoint. |
| `visual_omlx.py` | The **oMLX twin** of both `visual_ollama.py` and `visual_ml_studio.py`: the same raw HTTP POST of a base64 image to oMLX's OpenAI‑compatible endpoint, asking both originals' questions. |
| `test_lmstudio_vision.py` | The smallest possible LM Studio vision call — one image, one sentence back. |
| `test_omlx_vision.py` | The **oMLX twin**: the same one‑image, one‑sentence call, with the `openai` SDK instead of the LM Studio SDK. |
| `find_beer.py` | **Object detection / grounding.** Asks several LM Studio vision models to return bounding boxes for objects in a photo, then draws them with Pillow — and compares models. |
| `find_beer_omlx.py` | The **oMLX twin**: the same photo, prompt and box drawing, comparing Qwen3.5‑2B (4‑bit and 8‑bit) and Qwen3.5‑4B on oMLX, thinking off so the reply is just the JSON. Saves `*-processed-omlx-<model>.jpg`. Needs `pip install openai python-dotenv pillow`. |
| `read_diagram.py` | Reads a diagram and returns a **Mermaid** description of its structure. |
| `read_diagram_omlx.py` | The **oMLX twin** of `read_diagram.py`. |
| `read_music.py` | Reads a music score and names the **key signature**. |
| `read_music_omlx.py` | The **oMLX twin** of `read_music.py`. |
| `read_ocr_menu.py` | **OCR** a menu/photo to text (the "document is a PDF/image" path into RAG). |
| `read_ocr_menu_omlx.py` | The **oMLX twin** of `read_ocr_menu.py`. |
| `ice_cream_flavours.py` | A quick **menu‑reading** demo: base64‑encode an ice‑cream menu photo (`data/IMG_3319.jpeg`, with `IMG_3320`/`IMG_3321` as alternatives) and ask Ollama's `qwen3.5:4b` vision model to list the flavours. |
| `ice_cream_flavours_omlx.py` | The **oMLX twin**: the same photo and prompt, still one raw HTTP POST, to oMLX instead of Ollama. |
| `read_pdf.py` | **Lab: read a PDF page, then ask about its chart.** Renders ONE page of the Berlin Senate's investor presentation (August 2026, from berlin.de), which is published page for page in **German** (*Land Berlin – Investorenpräsentation*, `data/land-berlin-2026.pdf`, the default) and **English** (*State of Berlin – Investor Presentation*, `data/state-of-berlin-2026.pdf`). The `qwen3.5:4b` vision model transcribes the page to Markdown in the page's own language (tables become Markdown tables), saved as `berlin-<de|en>-page-<n>.md`. Then ask about the chart in English, or any language. Page numbers are the ones printed on each page. Page 8 (the default) is a line chart of real GDP growth, Berlin vs Germany, 2007–2025; page 9 is a chart of annual investment in Berlin start-ups: `python read_pdf.py 9`. The English edition of the same page: `python read_pdf.py 8 en`. About 10–15 s a page. Needs `pip install pymupdf`. |
| `read_pdf_omlx.py` | The **oMLX twin** of `read_pdf.py`: same page, prompts and arguments (`python read_pdf_omlx.py 8 en`), but the page image goes as a base64 data URL to oMLX's OpenAI‑compatible endpoint on `:8000` (`mlx-community--Qwen3.5-4B-MLX-4bit`, thinking off via `enable_thinking=False`). Saves `berlin-<de|en>-page-<n>-omlx.md` so it doesn't overwrite the Ollama run. Needs `OMLX_API_KEY` in the environment or the repo-root `.env`, and `pip install pymupdf openai python-dotenv`. |

## 2. RAG — Retrieval‑Augmented Generation

The heart of Day 2. Every `rag_*.py` shares one pipeline: **chunk** the text →
**embed** the chunks → embed the **query** → **cosine‑similarity** top‑k → pass the
query plus *only the retrieved context* to the LLM. They differ only by storage /
embedding backend, so you can compare approaches.

| Script | What it shows |
|---|---|
| `rag_alice_simple.py` | The minimal RAG loop on *Alice in Wonderland*, embeddings via `embeddinggemma` (Ollama). Start here. |
| `rag_alice_simple_omlx.py` | The **oMLX twin**: the same loop, with EmbeddingGemma and Qwen3.5‑4B both served by oMLX (still raw `requests`, so you see every HTTP call). |
| `rag_alice_in_wonderland.py` | The same, refactored into a reusable in‑memory `Document`/RAG structure. |
| `rag_alice_in_wonderland_omlx.py` | The **oMLX twin** of `rag_alice_in_wonderland.py`; the chunks are embedded in batches. |
| `rag_alice_in_wonderland_chromadb.py` | Stores the chunk embeddings in a **Chroma** vector database instead of memory — so you don't re‑embed every run. |
| `rag_alice_in_wonderland_chromadb_omlx.py` | The **oMLX twin**, with its own store (`chroma_db_omlx/`, collection `text_chunks_omlx`) so it never mixes vectors with the Ollama version. Needs `pip install chromadb`. |
| `rag_alice_in_wonderland_transformers.py` | Runs the embedding model **locally via `transformers`** (no Ollama embed endpoint) — usually faster. |
| `rag_alice_in_wonderland_transformers_omlx.py` | The **oMLX twin**: the embeddings stay local in `transformers` (that is the point of the demo); only the answers move from Ollama to oMLX. |
| `rag_grimm_fairy_tales.py` | The Alice pipeline pointed at *Grimm's Fairy Tales* (works in multiple languages — there's a German copy in `data/`). |
| `rag_grimm_fairy_tales_omlx.py` | The **oMLX twin**: EmbeddingGemma and Qwen3.5‑4B both on oMLX. |
| `rag_grimm_fairy_tales_groq.py` | The cloud twin: `sentence-transformers` embeddings + **Groq** (`llama-3.3-70b-versatile`) for generation. Keep it in sync with the local version. |
| `grimm_fairy_tales_rag_demo.py` | A richer demo using a **`Qwen3-Embedding`** model (batched, 512‑dim, with a reranking‑style top‑k) and `rich` output. |
| `grimm_fairy_tales_rag_demo_omlx.py` | The **oMLX twin**: the same Qwen3‑Embedding search and Qwen3.5 answers, both served by oMLX, so there is no `torch` to install. Needs `pip install openai python-dotenv numpy rich tiktoken`. |
| `alice_in_one_go.py` | The **CAG** counterpoint (cache‑augmented generation): the corpus is small enough to fit in a 64k context, so skip retrieval and feed the *whole book* to the model. |
| `alice_in_one_go_omlx.py` | The **oMLX twin**. oMLX has no per‑request `num_ctx`: it serves the model's native context (262k tokens for Qwen3.5‑4B), so the ~38k‑token book fits. |
| `kv_cache_vs_rag.py` | **Lab: KV-cache vs RAG.** Four questions about *Alice* on the class Spark, timing the **first token**: the whole book with a per-person tag at the start (so your first question really is uncached: ~6.5 s alone), then the same book again (read from the cache: ~0.9 s); a timestamp added at the **start** invalidates the cache, the same timestamp at the **end** doesn't; RAG with 4 chunks (~0.25 s, but it can miss things). With the whole room running it at once the Spark is shared, so expect several times longer. Needs `SPARK_API_KEY` in `.env` and `ollama pull all-minilm`. |
| `kv_cache_vs_rag_omlx.py` | The **oMLX twin**: the same lab on your own Mac, with oMLX's prefix cache and all‑MiniLM embeddings instead of the Spark and Ollama. Uses chapters I–III (~8k tokens) by default; `--whole-book` sends all ~38k tokens (~35 s to read the first time). No Spark key needed. |

## 3. Summarisation & Data Extraction

Summarisation taken to the extreme *is* data extraction. These show both, plus
the common "scrape a page, then extract" pattern.

| Script | What it shows |
|---|---|
| `data_extraction_ollama.py` | `summarise()` and `extract()` helpers over a sample model card — the basic summarise/extract pattern. |
| `data_extraction_groq.py` | The cloud twin (Groq `llama-3.3-70b-versatile`). |
| `data_extraction_omlx.py` | The **oMLX twin** (Qwen3.5‑4B on oMLX, thinking off). |
| `scrape.py` | A generic **BeautifulSoup** web scraper — the raw‑text source for extraction. |
| `scrape_gdpr_article.py` | Scrapes a GDPR article from gdpr-info.eu — the basis for the "summarise each article" exercise (see also `vibe/`). |
| `scrape_gdpr_articles_batch_omlx.py` | The **oMLX twin** of `scrape_gdpr_articles_batch.py`: summarises GDPR articles 20–25 with Qwen3.5‑4B on oMLX, from the saved pages in `data/gdpr-info/` (`--live` fetches them from gdpr-info.eu), into `data/gdpr_articles_20_25_omlx.json`. Thinking is off here, unlike the original: with it on, one article took over eight minutes. Needs `pip install requests beautifulsoup4 python-dotenv`. |
| `gdpr_viewer_omlx.py` | The **oMLX twin** of `gdpr_viewer.py`: the same Flask viewer on <http://localhost:7004>, showing the summaries from `scrape_gdpr_articles_batch_omlx.py` (run that first). Makes no model calls itself. Needs `pip install flask`. |
| `kaggle_summary_complete.py` | End‑to‑end: download a Kaggle dataset, run sentiment over it with `TextBlob`, and plot the results with matplotlib. |

## 4. Sentiment

Sentiment is just extraction where the thing you extract is a *mood*.

| Script | What it shows |
|---|---|
| `analyse_sentiment_01.py` | One‑word sentiment from a local model, with light "old‑fashioned" string cleanup to make a small model reliable. |
| `analyse_sentiment_01_omlx.py` | The **oMLX twin**: the same loop and cleanup, as a raw HTTP call to oMLX. |
| `analyse_sentiment_02.py` | The same with slightly different post‑processing — compare robustness. |
| `analyse_sentiment_02_omlx.py` | The **oMLX twin** of `analyse_sentiment_02.py`. |
| `analyse_sentiment_kaggle.py` | Run sentiment over a real dataset (Trump tweets from Kaggle). Use a `time.sleep(1)` if you point this at anything remote, to avoid rate‑limits. |

## 5. Structured (JSON) output

You get far more reliable output asking for **JSON** than free‑form text.

| Script | What it shows |
|---|---|
| `formatted_response_example.py` | Ollama's `format="json"` mode — ask for numbers 1‑10 in nine languages and parse the result. |
| `formatted_response_example_omlx.py` | The **oMLX twin**: the same JSON request, with JSON mode as the OpenAI‑style `response_format`. |
| `formatted_response_streaming.py` | The same, **streamed** token‑by‑token. |
| `formatted_response_streaming_omlx.py` | The **oMLX twin** of the streamed version. |
| `formatted_response_streaming_example_omlx.py` | The **oMLX twin** of `formatted_response_streaming_example.py`: streams the raw JSON, then parses it. |

## 6. Code generation → Vibe coding

LLMs are good at code (SQL, regex, whole programs). The deck demos Claude Code;
the local scripts here show generating, running and **testing** code, and the
high‑value special case of **SQL generation** from a schema.

Want the agentic "vibe coding" experience entirely on a local model? No API key
required:

- **[CODEX_OLLAMA_QUICKSTART.md](CODEX_OLLAMA_QUICKSTART.md)** — OpenAI's Codex
  CLI on an Ollama model. The easy path (no proxy). Start here.
- **[CLAUDE_CODE_OLLAMA_QUICKSTART.md](CLAUDE_CODE_OLLAMA_QUICKSTART.md)** — Claude
  Code on an Ollama model (one extra step: a tiny local translator).

| Script | What it shows |
|---|---|
| `payroll.py` | Downloads a payroll dataset from Kaggle and loads it into a **SQLite** database — the setup step. |
| `payroll2.py` | Feeds the **table schema plus a few sample rows** to a local LLM, extracts the generated SQL from the reply, then executes it. Giving the model sample data dramatically improves the SQL. |
| `payroll2_omlx.py` | The **oMLX twin**: the same schema‑plus‑samples prompt, as a raw HTTP call to oMLX. Run `payroll.py` first. |
| `payroll2_groq.py` | The cloud twin (Groq). |
| `vibe/` | The **vibe‑coding exercise** — `instruct.txt` is the natural‑language brief ("scrape the GDPR articles, build a site to browse them…") you hand to an agentic coding tool. |

## 7. Tool / function calling

Letting the model call your functions — the web, a database, the time, anything.

| Script | What it shows |
|---|---|
| `simple_tool_call.py` | The minimal pattern: a `tools` JSON schema + an `available_functions` dict — detect `tool_calls`, dispatch with the parsed arguments, append a `role:"tool"` message, re‑call. The currency‑conversion example from the slides. |
| `simple_tool_call_omlx.py` | The **oMLX twin**: the same raw‑HTTP tool loop (`role:"tool"` reply with its `tool_call_id`) against oMLX's `/v1/chat/completions` with Qwen3.5‑4B. |
| `ollama_function_support.py` | Scales that into a **benchmark harness** that iterates over installed Ollama models and scores their function‑calling (success rate, latency, σ) — because tool‑calling is non‑deterministic and not every model can do it. |
| `omlx_function_support.py` | The **oMLX twin**: the same scorecard over the chat models oMLX serves (up to 10 GB by default; `--include-big` adds the big ones; pass model ids to test a subset). A full run on four small models takes about 50 minutes; on this Mac Qwen3.5‑4B scored 90/100, Qwen3.5‑2B 74, Qwen3.5‑0.8B 50 and Granite‑4.1‑3B 41. Needs `pip install openai requests tabulate python-dotenv`. |

## 8. MCP — Model Context Protocol

The standardised way to publish and consume tools. See **[MCP/README.md](MCP/README.md)**
for full detail.

| Script | What it shows |
|---|---|
| `MCP/mcp_server.py` | An MCP **server** exposing two tools (`get_country_info`, `get_weather`) over stdio JSON‑RPC. |
| `MCP/test_mcp_client_ollama.py` | An MCP **client** running an agentic loop against the server with a local Ollama model — e.g. "what's the weather in the capital of Germany?" chains two tool calls. |
| `MCP/test_mcp_client_omlx.py` | The **oMLX twin**: the same MCP loop, with the MCP tools converted to OpenAI `tools` and Qwen3.5‑4B on oMLX. Needs `pip install "mcp<2" httpx requests python-dotenv` (`mcp` 2.x dropped the API `mcp_server.py` uses). |

## 9. Beyond — speech, and a note on agents

| Script | What it shows |
|---|---|
| `chatterbox_tts.py` | **Text → speech** with voice cloning via MLX (`mlx-audio`) — the "goodbye from Ricky" TTS demo. Apple‑Silicon only. Sample outputs live in `data/` (`ricky-berlin-ai-full.mp3`, `klopp-de*.mp3`). |
| `chatterbox_tts_omlx.py` | The **oMLX twin** of both `chatterbox_tts.py` (MLX) and `chatterbox_turbo_tts.py` (PyTorch): the same Morgan Freeman clone with Chatterbox‑Turbo (fp16; the quantized versions lack the speaker‑encoder weights, so they miss the voice) served by oMLX, so nothing loads in the script. Whisper on oMLX first supplies the sample's transcript, which oMLX wants along with the clip. Writes `day-2/output/demo_omlx.wav` and plays it. Needs `pip install openai python-dotenv sounddevice numpy`. |
| `clone_voice.py` | **Clone a voice** from a 5–15 s sample (Roosevelt by default) with Chatterbox on the class Spark: plays it out loud and saves it to `output/`. Windows, Mac or Linux; nothing to download (`pip install openai sounddevice numpy`). |
| `clone_voice_omlx.py` | The **oMLX twin**: the Roosevelt clone with Chatterbox‑Turbo on your own Mac instead of the Spark. Turbo is English‑only, so `LANGUAGE` has no effect ("Guten Morgen" gets English pronunciation), and it has no exaggeration or cfg settings. Writes `day-2/output/cloned_<voice>_omlx.wav` and plays it. Needs `pip install openai python-dotenv sounddevice numpy`. |
| `chatterbox_turbo_tts.py` | The same voice-cloning demo with PyTorch (`pip install chatterbox-tts`), for **Windows and Linux** (also runs on a Mac). Uses an NVIDIA GPU if there is one, otherwise the CPU. Copy the weights from the USB drive's `6-voice-windows-linux` folder first; otherwise it downloads 3.8 GB. |

**Music generation (slide: YuE2).** Two 39-second songs written for the class, `data/yue2-open-weights-en.mp3` and `data/yue2-offene-gewichte-de.mp3`. Both were made on a laptop with [YuE2-3B](https://huggingface.co/m-a-p) (8-bit MLX), which writes a sheet-music score first and then sings it, vocals and band together: about 13 s per song on an M5 Max. Style prompt: *English (or German), electronic, synth-pop, robotic male vocal, vocoder, analog synthesizers, drum machine, minimal, motorik, deadpan, 120 BPM*. The model is CC BY-NC 4.0 (open weights, non-commercial use only), and the files are tagged as AI-generated. The same model runs on iPhone and iPad in the YuE2 Studio app: github.com/Incept5-AI/YuE2.

The closing slides (Embabel, Open‑Claw, Hermes Agent, running an agent 24/7 on a
Raspberry Pi, email/home access) are **conceptual** — no scripts in this folder.
The takeaway: *the future is small, local, open‑weight models you own and control.*

---

## Suggested running order

1. `visual_ollama.py` — recap: a model reads an image.
2. `rag_alice_simple.py` → `rag_alice_in_wonderland.py` → `_chromadb.py` — build RAG, then add a vector DB.
3. `alice_in_one_go.py` — compare RAG with the "fits in context" (CAG) approach.
4. `data_extraction_ollama.py` → `scrape_gdpr_article.py` — summarise and extract.
5. `analyse_sentiment_01.py` — extraction as sentiment.
6. `formatted_response_example.py` — make output reliable with JSON.
7. `payroll.py` → `payroll2.py` — generate and run SQL from a schema.
8. `simple_tool_call.py` → `ollama_function_support.py` — tool calling, then benchmark it.
9. `MCP/test_mcp_client_ollama.py` — the same idea standardised over MCP.

Without Ollama or LM Studio, run the `*_omlx.py` twin of each step instead (`visual_omlx.py`, `rag_alice_simple_omlx.py`, …).

## 8. Agents working together

| Script | What it shows |
|---|---|
| `persona_debate.py` | **Lab: persona debate.** Three to five characters, each with a back-story, a personality and its own corpus as memory, debate a question over two rounds, then each votes for the most convincing *other* speaker. Pick a `SET`: **film**, the default (Darth Vader, Mary Poppins, Captain Kirk and the Terminator on breaking the rules; scripts are copyrighted, so their memories are a few lines written for the lab), **philosophers** (Socrates, Machiavelli, Nietzsche on *"Is it better to be feared or loved?"*, each reading their own book), **famous** (Oscar Wilde, Benjamin Franklin, Mark Twain on money and happiness) or the original **literary** set (Alice, the Hatter, Little Red-Cap). The philosophers' and famous people's texts are public-domain Project Gutenberg books in `data/`. Runs on the class Spark (`SPARK_API_KEY` in `.env`); switch `base_url`/`MODEL` to use Ollama. Change the question, swap a persona, or add your own. |
| `persona_debate_omlx.py` | The **oMLX twin**: the same sets, debate and vote on Qwen3.5‑4B via oMLX, no Spark key needed. `OMLX_MODEL=Qwen3.8-27B-oQ4e-mtp python persona_debate_omlx.py` uses a bigger model you have in oMLX (sharper debates, much slower). |

| `browser_tool_call.py` + `profile_markdown.py` | **Demo: an agent with a browser.** Qwen3.5-4B (Ollama) gets two tools that drive your own logged-in **Ego Lite** browser (https://lite.ego.app). `read_open_profile(url?)` reads a LinkedIn profile (the one open in Ego Lite, or a given address) in its own tab: it scrolls until everything has loaded, clicks every "… more", presses "Show more results", reads the "Show all" pages, opens the Contact info pop-up and the Interests tabs, all at a person's pace with small random pauses. No LinkedIn API calls: only what the page shows. It saves three files named after the profile address: `output/<profile>.json` in **Proxycurl's profile format** (as in the 2024 archive, with endorsements, recommendation authors and post stats under `extra`), `<profile>.md` laid out from that JSON, and `<profile>.jpg`. `python profile_markdown.py some.json` re-renders the Markdown and also works on old Proxycurl files. `read_page(url)` reads any page. A profile is someone's personal data and LinkedIn's terms forbid automated scraping: ask first, keep the files to yourself, delete them after. |
| `browser_tool_call_omlx.py` | The **oMLX twin** of `browser_tool_call.py`: the same Ego Lite browser agent and tools, with Qwen3.5‑4B on oMLX. `read_page` and the answer go to `output/page-omlx.md` and `output/answer-omlx.md`, so they don't overwrite the Ollama run's files. Needs `pip install requests python-dotenv`. |
