import os

import requests

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set the OMLX_API_KEY environment variable to your oMLX API key.")

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "mlx-community--Qwen3.5-4B-MLX-4bit",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 256,      # Ollama's num_predict
        "seed": 42,
        # No num_ctx: oMLX sizes the context itself, up to the model's max_model_len (see /v1/models)
    },
)

data = response.json()

if "choices" not in data:
    raise SystemExit(
        "oMLX did not return a completion. Response was:\n"
        f"{data}\n\n"
        "Make sure the server is running and the model name is correct "
        "(see available models at http://127.0.0.1:8000/v1/models)."
    )

print(data["choices"][0]["message"]["content"].strip())
