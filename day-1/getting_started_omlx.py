import os

import requests

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set the OMLX_API_KEY environment variable to your oMLX API key.")

response = requests.post("http://127.0.0.1:8000/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "Qwen3.8-27B-oQ4e-mtp",
            "messages": [{"role": "user", "content": "Hello"}]}
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
