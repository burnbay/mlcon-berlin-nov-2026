#!/usr/bin/env python3
"""The oMLX twin of logit_probabilities.py: the same Gradio token-probability explorer (Chat and Generate
modes, top-5 next tokens, pie chart, click a candidate to inject it), but served by oMLX: the probabilities
come from `logprobs` / `top_logprobs` on /v1/chat/completions instead of llama.cpp's logits.

How it maps onto oMLX:
- The exact raw prompt (the hand-built ChatML string in Chat mode, the bare text in Generate mode) is sent as
  a single message through a pass-through chat template, so nothing is wrapped around it.
- Each step is a one-token completion with `top_logprobs=20` (the most oMLX returns) and `ignore_eos`, so an
  end-of-turn token shows up as a candidate instead of ending the request with no logprobs.
- oMLX's log-probs are at temperature 1, so the temperature is applied here, over the 20 candidates: exact
  at 1.0, an estimate otherwise (the rest of the vocabulary isn't returned).
- oMLX takes text, not token ids, so the continuation is held as text. A candidate that is only part of a
  character (a byte-fallback token) can't be appended on its own, so its button is disabled.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache (or OMLX_MODEL set
to another oMLX model id), and OMLX_API_KEY in the environment or the repo-root .env.
Needs `pip install openai python-dotenv gradio plotly`.
"""
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

# Qwen3.8-27B-oQ4e-mtp is the sharper but slower option
DEFAULT_MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"
DEFAULT_SYSTEM_PROMPT = "You are a Yoda assistant, you always reply like Yoda and often mention the force. You are wise with your words."
DEFAULT_USER_PROMPT = "Where is Westminster Abbey?"
DEFAULT_GENERATE_PROMPT = "The cat shat on the"

# Fixed display size: top 5 candidate tokens + an "other" wedge for the remainder.
NUM_TOKENS_DISPLAYED = 5
# oMLX rejects top_logprobs above 20.
TOP_LOGPROBS = 20
# Renders the first message verbatim, so the prompt reaches the model exactly as built here.
RAW_TEMPLATE = "{{ messages[0]['content'] }}"

# Qwen3-series recommended sampling for *non-thinking* mode (the mode this app forces).
# Source: Qwen team guidance for Qwen3 chat models; presence/repetition penalties off by default.
QWEN3_NO_THINK_DEFAULTS = dict(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repeat_penalty=1.0,
    num_tokens=NUM_TOKENS_DISPLAYED,
)


@dataclass
class TokenCandidate:
    """One next-token candidate.

    `raw` is the token's exact bytes (oMLX's `bytes` field). A byte-fallback token
    carries only *part* of a multi-byte character — each Arabic-Indic digit
    (U+0660-U+0669) is two such tokens in Qwen3.5, since the vocab has no merge
    for them — so decoding it on its own yields U+FFFD.

    `display` is what the UI shows: the decoded text, or llama.cpp-style
    `<0xD9>` hex when the token isn't valid UTF-8 by itself.
    """
    raw: bytes
    display: str
    probability: float
    logprob: float


@dataclass
class InferenceConfig:
    temperature: float = QWEN3_NO_THINK_DEFAULTS["temperature"]
    top_p: float = QWEN3_NO_THINK_DEFAULTS["top_p"]
    top_k: int = QWEN3_NO_THINK_DEFAULTS["top_k"]
    repeat_penalty: float = QWEN3_NO_THINK_DEFAULTS["repeat_penalty"]
    num_tokens: int = QWEN3_NO_THINK_DEFAULTS["num_tokens"]


def describe_token_bytes(raw: bytes) -> str:
    """Decoded text for `raw`, or `<0xD9>`-style hex if it isn't valid UTF-8 alone."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"<0x{b:02X}>" for b in raw)


def build_chat_prompt(system_prompt: str, user_prompt: str, assistant_prefix: str = "") -> str:
    """Qwen3 ChatML prompt with thinking mode forcibly disabled.

    Two complementary mechanisms (both required for reliable no-think behavior):
      1. `/no_think` directive appended to the system prompt.
      2. Pre-seeded empty <think>\\n\\n</think>\\n\\n block at the start of the assistant turn,
         so the very first token the model emits is the actual answer.
    """
    system_text = system_prompt.strip() if system_prompt.strip() else "You are a helpful assistant."
    return (
        f"<|im_start|>system\n{system_text} /no_think<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_prefix}"
    )


class TokenProbabilityAnalyzer:
    # Special tokens we treat as "stop signals". oMLX only returns the top 20
    # candidates, so we report their probability whenever one makes the top 20,
    # and an upper bound (below the 20th candidate) when none does.
    STOP_TOKEN_LITERALS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>", "</s>")

    def __init__(self, model_id: str):
        served = [m.id for m in client.models.list()]
        if model_id not in served:
            raise ValueError(f"oMLX doesn't serve '{model_id}' (see http://127.0.0.1:8000/v1/models)")
        self.model = model_id
        self.debug_info = ""
        self.last_prompt = ""
        self.last_stop_probs: List[Tuple[str, float]] = []  # [(literal, prob)]
        self.last_floor: Optional[float] = None  # probability of the 20th candidate

    def analyze_prompt(self, prompt: str, config: InferenceConfig) -> List[TokenCandidate]:
        """Return the most likely next-token candidates for `prompt`, sent verbatim."""
        self.last_prompt = prompt
        self.debug_info = f"Prompt: {len(prompt)} chars\n"
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
            extra_body={
                "chat_template_kwargs": {"chat_template": RAW_TEMPLATE},
                "ignore_eos": True,
                # Applied by oMLX before the log-probs are computed, like the original's penalty on the logits
                "repetition_penalty": config.repeat_penalty,
            },
        )
        self.debug_info += f"→ {response.usage.prompt_tokens} tokens\n"
        choice = response.choices[0]
        steps = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []
        if not steps:
            self.debug_info += f"oMLX returned no logprobs (finish_reason={choice.finish_reason})\n"
            return []
        return self._top_tokens_from_logprobs(steps[0].top_logprobs, config)

    def _top_tokens_from_logprobs(self, top, config: InferenceConfig) -> List[TokenCandidate]:
        top = sorted(top, key=lambda e: -e.logprob)
        logprobs = [e.logprob for e in top]

        # Temperature T turns each log-prob l into l/T before renormalising. The 20
        # candidates are rescaled exactly; the tail oMLX doesn't return is treated as
        # tokens sitting at the 20th candidate's log-prob (exact at T=1, an estimate
        # otherwise). Done in log space so a tiny T doesn't overflow.
        temp = max(config.temperature, 1e-6)
        floor = logprobs[-1]
        terms = [lp / temp for lp in logprobs]
        tail = 1.0 - sum(math.exp(lp) for lp in logprobs)
        if tail > 0:
            terms.append(math.log(tail) - floor + floor / temp)
        peak = max(terms)
        log_z = peak + math.log(sum(math.exp(t - peak) for t in terms))

        candidates = []
        for entry in top:
            raw = bytes(entry.bytes) if entry.bytes is not None else entry.token.encode("utf-8")
            candidates.append(TokenCandidate(raw, describe_token_bytes(raw),
                                             math.exp(entry.logprob / temp - log_z), entry.logprob))

        # Capture stop-token probabilities BEFORE trimming to the top 5, so the UI can
        # show them even when EOS is ranked below the displayed candidates (typical
        # mid-paragraph in Generate mode — exactly the "why doesn't it stop?" moment).
        self.last_stop_probs = [(c.display, c.probability) for c in candidates
                                if c.display in self.STOP_TOKEN_LITERALS]
        self.last_floor = candidates[-1].probability

        # Always surface the top `num_tokens` candidates by *raw* probability.
        # We deliberately do NOT use top_p/top_k as display filters: with an
        # aggressive top_p (e.g. 0.8 when the model is confident) the survivor
        # set collapses to a single token, hiding the long tail this tool exists
        # to show. top_p/top_k remain in the UI as sampler-reference values.
        return candidates[:config.num_tokens]


current_analyzer: Optional[TokenProbabilityAnalyzer] = None
last_analysis_results: Optional[List[TokenCandidate]] = None


def load_model(model_id: str) -> str:
    global current_analyzer, last_analysis_results
    if not model_id.strip():
        return "Error: Model id cannot be empty"
    try:
        current_analyzer = TokenProbabilityAnalyzer(model_id.strip())
        last_analysis_results = None
        return f"✅ Using oMLX model: {model_id.strip()}"
    except Exception as e:
        return f"❌ Error: {e}"


STOP_TOKENS = {"</s>", "<|endoftext|>", "<|im_end|>", "<|end|>", "<|eot_id|>", "<eos>"}
WHITESPACE_CONTROLS = {"\n", "\t", "\r"}


def is_stop_token(candidate: TokenCandidate) -> bool:
    if not candidate.raw:
        return True
    token = candidate.display
    if token in STOP_TOKENS:
        return True
    # Single-char unprintable control (excluding ordinary whitespace) counts as a stop;
    # newlines and tabs are normal generation tokens and must not disable the flow.
    if len(token) == 1 and ord(token) < 32 and token not in WHITESPACE_CONTROLS:
        return True
    return False


def can_inject(candidate: TokenCandidate) -> bool:
    """oMLX takes text, so only a candidate that is whole characters by itself can be appended."""
    if is_stop_token(candidate):
        return False
    try:
        candidate.raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _disabled_buttons():
    """Default 5-button state shown before any analysis exists."""
    return [gr.Button(value=f"#{i + 1}", interactive=False) for i in range(NUM_TOKENS_DISPLAYED)]


def _button_updates_for(token_probs):
    """Label each candidate button with its rank + token preview, and disable
    buttons whose token is a stop token (clicking them would terminate) or only
    part of a character (oMLX can't be sent half a character)."""
    updates = []
    for i in range(NUM_TOKENS_DISPLAYED):
        if i < len(token_probs):
            candidate = token_probs[i]
            preview = repr(candidate.display)
            if len(preview) > 14:
                preview = preview[:11] + "…"
            label = f"#{i + 1} {preview}"
            updates.append(gr.Button(value=label, interactive=can_inject(candidate)))
        else:
            updates.append(gr.Button(value=f"#{i + 1}", interactive=False))
    return updates


def _analyze_response(token_probs, formatted_prompt_text):
    if not token_probs:
        msg = (
            "**No tokens returned.**\n\n"
            f"```\n{current_analyzer.debug_info if current_analyzer else ''}```"
        )
        return (msg, None, *_disabled_buttons(), formatted_prompt_text)
    stop_probs = list(current_analyzer.last_stop_probs) if current_analyzer else []
    floor = current_analyzer.last_floor if current_analyzer else None
    body = format_top_tokens(token_probs, stop_probs, floor)
    chart = create_pie_chart(token_probs)
    return (body, chart, *_button_updates_for(token_probs), formatted_prompt_text)


def analyze_tokens(mode, system_prompt, user_prompt, generate_prompt,
                   temperature, top_p, top_k, repeat_penalty):
    global current_analyzer, last_analysis_results

    if current_analyzer is None:
        return ("⚠️ No model selected — open the **Model Configuration** tab to pick one.",
                None, *_disabled_buttons(), "")

    active_prompt = user_prompt if mode == "Chat" else generate_prompt
    if not active_prompt or not active_prompt.strip():
        return ("⚠️ Prompt is empty — type something before clicking Analyze.",
                None, *_disabled_buttons(), "")

    config = InferenceConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        num_tokens=NUM_TOKENS_DISPLAYED,
    )

    try:
        prompt = build_chat_prompt(system_prompt, user_prompt) if mode == "Chat" else generate_prompt
        token_probs = current_analyzer.analyze_prompt(prompt, config)
        last_analysis_results = token_probs
        return _analyze_response(token_probs, current_analyzer.last_prompt)
    except Exception as e:
        import traceback
        return (f"Error: {e}\n\n```\n{traceback.format_exc()}```",
                None, *_disabled_buttons(), "")


def add_token_by_rank(rank, mode, system_prompt, user_prompt, generate_prompt, chat_response,
                      temperature, top_p, top_k, repeat_penalty):
    """Inject the candidate at 1-based `rank` (1..NUM_TOKENS_DISPLAYED) and re-analyze."""
    global current_analyzer, last_analysis_results

    if (current_analyzer is None or not last_analysis_results
            or rank < 1 or rank > len(last_analysis_results)
            or not can_inject(last_analysis_results[rank - 1])):
        return ("Run **Analyze** first.", None, *_disabled_buttons(),
                generate_prompt, chat_response, "")

    config = InferenceConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        num_tokens=NUM_TOKENS_DISPLAYED,
    )

    try:
        # The continuation is just the text in the box (hand edits included) plus the
        # chosen token. oMLX re-tokenises it on every step, which almost always gives
        # back the same tokens the model produced.
        visible = chat_response if mode == "Chat" else generate_prompt
        grown_text = visible + last_analysis_results[rank - 1].display

        # In Chat mode the assistant prefix is the last thing build_chat_prompt emits,
        # so the model continues straight from the grown response.
        if mode == "Chat":
            new_chat_response = grown_text
            prompt = build_chat_prompt(system_prompt, user_prompt, assistant_prefix=grown_text)
            new_generate_prompt = generate_prompt
        else:
            new_generate_prompt = grown_text
            prompt = grown_text
            new_chat_response = chat_response

        token_probs = current_analyzer.analyze_prompt(prompt, config)
        last_analysis_results = token_probs
        body, chart, *btn_and_prompt = _analyze_response(token_probs, current_analyzer.last_prompt)
        *btn_updates, prompt_view = btn_and_prompt
        return (body, chart, *btn_updates, new_generate_prompt, new_chat_response, prompt_view)
    except Exception as e:
        import traceback
        return (f"Error: {e}\n\n```\n{traceback.format_exc()}```", None,
                *_disabled_buttons(), generate_prompt, chat_response, "")


def reset_state(mode):
    """Clear chat response / generate prompt back to defaults and wipe analysis state."""
    global last_analysis_results
    last_analysis_results = None
    if mode == "Chat":
        return (gr.update(value=""), gr.update(value=DEFAULT_USER_PROMPT),
                "Cleared. Click **Analyze** to start again.",
                None, *_disabled_buttons(), "")
    return (gr.update(value=""), gr.update(value=DEFAULT_GENERATE_PROMPT),
            "Cleared. Click **Analyze** to start again.",
            None, *_disabled_buttons(), "")


def _format_pct(prob: float) -> str:
    pct = prob * 100
    return f"{pct:.4f}%" if pct >= 0.0001 else f"{pct:.2e}%"


def format_top_tokens(token_probs: List[TokenCandidate],
                      stop_probs: Optional[List[Tuple[str, float]]] = None,
                      floor: Optional[float] = None) -> str:
    lines = [f"**Top {len(token_probs)} candidate tokens**", ""]
    lines.append("| # | Token | Probability | Log-prob (T=1) |")
    lines.append("|---:|---|---:|---:|")
    for i, candidate in enumerate(token_probs, 1):
        marker = "🛑 " if is_stop_token(candidate) else ""
        lines.append(f"| {i} | {marker}`{repr(candidate.display)}` "
                     f"| {candidate.probability * 100:.2f}% | {candidate.logprob:.3f} |")
    # Combined stop-token row directly under the top-5, even when EOS is ranked
    # below them — the canonical "why doesn't it stop?" readout. Outside oMLX's
    # top 20 only an upper bound is known.
    if stop_probs:
        cum = sum(p for _, p in stop_probs)
        lines.append(f"| 🛑 | _stop tokens (combined)_ | {_format_pct(cum)} | — |")
    elif floor is not None:
        lines.append(f"| 🛑 | _stop tokens (not in the top {TOP_LOGPROBS})_ | < {_format_pct(floor)} | — |")
    total = sum(c.probability for c in token_probs) * 100
    lines.append("")
    lines.append(f"_Top {len(token_probs)} cover **{total:.2f}%** of the full distribution._")
    return "\n".join(lines)


def create_pie_chart(token_probs: List[TokenCandidate]):
    if not token_probs:
        return None

    labels, values, colors, hover = [], [], [], []
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899",
               "#06b6d4", "#84cc16", "#f43f5e", "#a855f7", "#14b8a6"]
    palette_idx = 0
    for candidate in token_probs:
        display = repr(candidate.display)
        if len(display) > 22:
            display = display[:19] + "…"
        pct = candidate.probability * 100
        if is_stop_token(candidate):
            labels.append(f"🛑 {display} ({pct:.2f}%)")
            colors.append("#ef4444")
        else:
            labels.append(f"{display} ({pct:.2f}%)")
            colors.append(palette[palette_idx % len(palette)])
            palette_idx += 1
        values.append(candidate.probability)
        hover.append(f"token: {repr(candidate.display)}<br>probability: {pct:.3f}%"
                     f"<br>log-prob (T=1): {candidate.logprob:.3f}")

    # The displayed tokens rarely cover 100% of the vocab — represent the rest
    # honestly with a grey "other" wedge so the pie shows the true distribution.
    shown = sum(values)
    if shown < 0.9995:
        remaining = 1.0 - shown
        labels.append(f"(other tokens — {remaining * 100:.2f}%)")
        values.append(remaining)
        colors.append("#e5e7eb")
        hover.append(f"all other vocab tokens<br>cumulative: {remaining * 100:.3f}%")

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, textinfo="none",
        marker=dict(colors=colors),
        hovertext=hover, hoverinfo="text",
        sort=False, direction="clockwise",
    )])
    fig.update_layout(
        title="Next-token probability distribution",
        height=310, font=dict(size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(font=dict(size=11)),
    )
    return fig


def create_gradio_interface():
    with gr.Blocks(title="Token Probability Analyzer") as app:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# 🔍 Token Probability Analyzer")
                gr.Markdown(
                    "Inspect what the model **would** sample next, token by token, served by oMLX. "
                    "Qwen3 thinking mode is forcibly **off** "
                    "(`/no_think` directive + pre-seeded empty `<think></think>` block)."
                )
            with gr.Column(scale=2):
                mode = gr.Radio(
                    choices=["Chat", "Generate"], value="Chat", label="Mode",
                    info="Chat: separate response window  •  Generate: continue raw text",
                )
                model_status = gr.Markdown("_No model selected_")

        with gr.Tab("Token Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    system_prompt = gr.Textbox(label="System Prompt (optional)", value=DEFAULT_SYSTEM_PROMPT, lines=2,
                                               placeholder="System instructions…", visible=True)
                    user_prompt = gr.Textbox(label="User Prompt", value=DEFAULT_USER_PROMPT, lines=2, visible=True)
                    chat_response = gr.Textbox(label="Assistant Response", value="", lines=4,
                                               placeholder="Response will grow here as you add tokens…", visible=True)
                    generate_prompt = gr.Textbox(label="Text to Continue", value=DEFAULT_GENERATE_PROMPT, lines=6,
                                                 placeholder="Text will grow here as you add tokens…", visible=False)

                    with gr.Accordion("Sampling parameters (Qwen3 no-think defaults)", open=False):
                        with gr.Row():
                            temperature = gr.Slider(0.01, 2.0, value=QWEN3_NO_THINK_DEFAULTS["temperature"],
                                                    label="Temperature")
                            top_p = gr.Slider(0.01, 1.0, value=QWEN3_NO_THINK_DEFAULTS["top_p"], label="Top-p")
                        with gr.Row():
                            top_k = gr.Slider(1, 100, value=QWEN3_NO_THINK_DEFAULTS["top_k"], step=1, label="Top-k")
                            repeat_penalty = gr.Slider(0.5, 2.0, value=QWEN3_NO_THINK_DEFAULTS["repeat_penalty"],
                                                       step=0.01, label="Repeat Penalty",
                                                       info="Qwen team recommends leaving this at 1.0.")

                    with gr.Row():
                        analyze_btn = gr.Button("🚀 Analyze", variant="primary")
                        reset_btn = gr.Button("🧹 Clear", variant="stop")

                    gr.Markdown("**Inject a candidate** (greedy = `#1`; pick any other to explore the path-not-taken):")
                    with gr.Row():
                        add_buttons = [
                            gr.Button(value=f"#{i + 1}", interactive=False,
                                      variant=("primary" if i == 0 else "secondary"))
                            for i in range(NUM_TOKENS_DISPLAYED)
                        ]

                    formatted_prompt_view = gr.Textbox(
                        label="Formatted prompt sent to model", lines=8, interactive=False,
                    )

                with gr.Column(scale=1):
                    chart = gr.Plot(label="")
                    results = gr.Markdown("Click **Analyze** to see results…")

        with gr.Tab("Model Configuration"):
            with gr.Row():
                with gr.Column():
                    # No context size or GPU layers here: oMLX manages both itself.
                    model_id = gr.Textbox(
                        label="oMLX Model",
                        value=os.environ.get("OMLX_MODEL", DEFAULT_MODEL),
                        placeholder="Model id from http://127.0.0.1:8000/v1/models",
                    )
                    load_btn = gr.Button("🔄 Use Model", variant="primary")
                with gr.Column():
                    load_status = gr.Textbox(label="Status", lines=4, interactive=False)

        def toggle_mode(m, current_user, current_generate):
            # Preserve whatever text the user has typed; only seed the mode's
            # default prompt when the target field is empty (Gradio doesn't
            # always re-render the initial `value` of a component that was
            # created with visible=False).
            user_val = current_user if (current_user and current_user.strip()) else DEFAULT_USER_PROMPT
            gen_val = current_generate if (current_generate and current_generate.strip()) else DEFAULT_GENERATE_PROMPT
            if m == "Chat":
                return (gr.update(visible=True),
                        gr.update(visible=True, value=user_val),
                        gr.update(visible=True),
                        gr.update(visible=False))
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True, value=gen_val))

        # queue=False + show_progress="hidden": this is a pure UI toggle, never
        # touching the model, so it shouldn't sit in the queue or show a spinner.
        mode.change(fn=toggle_mode, inputs=[mode, user_prompt, generate_prompt],
                    outputs=[system_prompt, user_prompt, chat_response, generate_prompt],
                    queue=False, show_progress="hidden")

        def load_and_update(m):
            msg = load_model(m)
            return msg, msg
        load_btn.click(fn=load_and_update, inputs=[model_id],
                       outputs=[load_status, model_status])

        analyze_btn.click(
            fn=analyze_tokens,
            inputs=[mode, system_prompt, user_prompt, generate_prompt,
                    temperature, top_p, top_k, repeat_penalty],
            outputs=[results, chart, *add_buttons, formatted_prompt_view],
        )

        # One click handler per candidate button; rank is bound at definition time
        # so each lambda captures its own value rather than the loop variable.
        for i, btn in enumerate(add_buttons):
            btn.click(
                fn=lambda *args, rank=i + 1: add_token_by_rank(rank, *args),
                inputs=[mode, system_prompt, user_prompt, generate_prompt, chat_response,
                        temperature, top_p, top_k, repeat_penalty],
                outputs=[results, chart, *add_buttons,
                         generate_prompt, chat_response, formatted_prompt_view],
            )

        reset_btn.click(
            fn=reset_state,
            inputs=[mode],
            outputs=[chat_response, generate_prompt, results, chart,
                     *add_buttons, formatted_prompt_view],
        )

    return app


def main():
    print("Starting Token Probability Analyzer…")
    model = os.environ.get("OMLX_MODEL", DEFAULT_MODEL)
    msg = load_model(model)
    if msg.startswith("✅"):
        print(msg)
    else:
        print(f"⚠️ {msg} — pick a model in the Model Configuration tab")

    port = os.environ.get("GRADIO_SERVER_PORT")
    app = create_gradio_interface()
    launch_kwargs = dict(server_name="127.0.0.1", share=False, theme=gr.themes.Soft())
    if port:
        launch_kwargs["server_port"] = int(port)
    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
