"""The oMLX twin of quantisation_taste_test.py: the same four prompts through Qwen3.5 at MLX 4-bit,
8-bit and BF16, with the size and tokens/sec of each.

    python day-1/quantisation_taste_test_omlx.py 0.8b 2b      (the default)
    python day-1/quantisation_taste_test_omlx.py 4b           (or 9b with 32 GB+), or give full model names
    python day-1/quantisation_taste_test_omlx.py mlx-community/Qwen3.5-4B-6bit

Download sizes (4bit / 8bit / bf16): 0.8b 0.6 / 1.0 / 1.7 GB · 2b 1.7 / 2.7 / 4.4 GB · 4b 3.0 / 5.1 / 9.1 GB
                                     9b 6.0 / 10 / 19 GB.  6-bit exists too, e.g. mlx-community/Qwen3.5-2B-6bit.

oMLX serves models straight from the Hugging Face cache, so `hf download mlx-community/Qwen3.5-2B-8bit`
is all it takes; oMLX sees new downloads after "Reload models" in its admin page (or `omlx restart`).
Needs the OMLX_API_KEY environment variable, as in getting_started_omlx.py.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

OMLX = "http://127.0.0.1:8000/v1"
HF_CACHE = Path(os.getenv("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
QUANTS = ["4bit", "8bit", "bf16"]
SIZES = sys.argv[1:] or ["0.8b", "2b"]
MODELS = [m for s in SIZES for m in ([f"mlx-community/Qwen3.5-{s.upper()}-{q}" for q in QUANTS]
                                     if re.fullmatch(r"[\d.]+b", s, re.IGNORECASE) else [s])]

PROMPTS = [
    "What is 17 x 24? Reply with just the number.",
    "Name five Berlin districts (Bezirke), comma separated.",
    "Translate into French: The early bird catches the worm.",
    "Write a haiku about the East Side Gallery.",
]

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set the OMLX_API_KEY environment variable to your oMLX API key.")
HEADERS = {"Authorization": f"Bearer {api_key}"}


def omlx_id(model):
    return model.replace("/", "--")  # oMLX names Hugging Face cache models "org--repo"


def size_gb(model):
    """Size of a complete download in the Hugging Face cache, or None."""
    for snap in (HF_CACHE / f"models--{omlx_id(model)}" / "snapshots").glob("*"):
        index = snap / "model.safetensors.index.json"  # lists every weight shard the model needs
        shards = (set(json.loads(index.read_text())["weight_map"].values()) if index.is_file()
                  else {f.name for f in snap.glob("*.safetensors")})
        if shards and all((snap / s).is_file() for s in shards):  # snapshots hold finished files only
            return sum(f.stat().st_size for f in snap.rglob("*") if f.is_file()) / 1e9
    return None


def generate(model, prompt, **extra):
    r = requests.post(f"{OMLX}/chat/completions", headers=HEADERS, timeout=600, json={
        "model": omlx_id(model), "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **extra})
    r.raise_for_status()
    return r.json()


installed = {m["id"] for m in requests.get(f"{OMLX}/models", headers=HEADERS, timeout=10).json()["data"]}

for model in MODELS:
    size = size_gb(model)
    if omlx_id(model) not in installed:
        hint = ("downloaded, but oMLX hasn't seen it yet: Reload models in the oMLX admin page, or `omlx restart`"
                if size else f"not downloaded, run: hf download {model}")
        print(f"\n=== {model}: {hint}")
        continue
    print(f"\n=== {model}" + (f" ({size:.1f} GB)" if size else ""))
    generate(model, "Hi", max_tokens=1)  # load it first so timings are fair
    for prompt in PROMPTS:
        start = time.time()
        r = generate(model, prompt, max_tokens=256)
        usage = r["usage"]
        # oMLX reports generation speed itself (prompt processing excluded), like Ollama's eval_duration
        tok_per_s = usage.get("generation_tokens_per_second") or usage["completion_tokens"] / (time.time() - start)
        print(f"  {prompt}\n    -> {r['choices'][0]['message']['content'].strip()}  "
              f"({time.time() - start:.1f} s, {tok_per_s:.0f} tok/s)")
