"""The oMLX twin of alice_in_one_go.py: the same CAG counterpoint to the RAG demos - no chunking or
retrieval, the whole cleaned book goes into one prompt with the question - but answered by Qwen3.5 on
oMLX's OpenAI-compatible API (still raw `requests`, thinking still off) instead of Ollama.

oMLX has no per-request `num_ctx`: it serves the model's native context (262,144 tokens for
Qwen3.5-4B) unless a cap is set per model in the oMLX admin page (~/.omlx/model_settings.json) or via
`max_context_window_policy` in ~/.omlx/settings.json. The book plus prompt is ~38k tokens, so it fits;
a prompt over the limit gets HTTP 400 "Prompt is too long".

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install requests python-dotenv`.
"""
from typing import List
import os
import requests
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
HEADERS = {"Authorization": f"Bearer {api_key}"}

LLM_MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"   # oMLX names Hugging Face cache models "org--repo"
HERE = Path(__file__).parent


class OMLXClient:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/v1"

    def generate_response(self, prompt: str) -> str:
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.05,
            "top_p": 0.85
        }

        print(f"Sending request to oMLX... (prompt length: {len(prompt)} chars)")
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=HEADERS, json=data, timeout=300)
            response.raise_for_status()
            body = response.json()
            result = body["choices"][0]["message"]["content"]
            print(f"Response received successfully! (prompt: {body['usage']['prompt_tokens']} tokens)")
            return result
        except requests.exceptions.Timeout:
            return "Error: Request timed out - document may be too large for model context"
        except requests.exceptions.HTTPError as e:
            # oMLX rejects an over-long prompt with a 400 whose body says why
            return f"Error: Network/API issue - {e} - {e.response.text}"
        except requests.exceptions.RequestException as e:
            return f"Error: Network/API issue - {e}"

class FullContextQA:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.omlx = OMLXClient()
        self.full_text = ""
        self._test_connection()
        self._load_document()

    def _test_connection(self):
        print("Testing oMLX connection...")
        try:
            test_response = self.omlx.generate_response("Say 'Hello' to test the connection.")
            if "Error:" not in test_response:
                print(f"Connection test successful: {test_response[:50]}...")
            else:
                print(f"Connection test failed: {test_response}")
                return False
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
        return True

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        if 'Contents' in text:
            match = re.search(r'CHAPTER I\.\s*\n\s*[\w\s-]+\n\s*\n', text)
            if match:
                text = text[match.end():]

        return text.strip()

    def _load_document(self):
        print(f"Loading document: {self.file_path.name}")

        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        self.full_text = self._clean_text(text)

        print(f"Document loaded: {len(self.full_text)} characters")
        print(f"Estimated tokens: ~{len(self.full_text) // 4}")

    def query(self, question: str) -> str:
        if not question.strip():
            return "Please provide a question."

        prompt = f"""Answer the question using the provided complete text.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the text doesn't contain enough information, say so clearly.
Focus on accuracy - only state what is supported by the text.

Question: {question}

Complete Text:
{self.full_text}

Answer:"""

        try:
            answer = self.omlx.generate_response(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    print("Full Context QA Demo System (whole book in context)")
    print("Make sure oMLX is running on port 8000\n")

    try:
        qa = FullContextQA(str(HERE / "data" / "alice_in_wonderland.txt"))
        print(f"\nFull-context system ready!\n")

        test_questions = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "What did the White Rabbit say when Alice first saw him?"
        ]

        print("Testing with first 3 questions:")
        print("=" * 70)

        for i, question in enumerate(test_questions, 1):
            print(f"Question {i}/3: {question}")
            print("-" * 50)

            answer = qa.query(question)
            print(f"A: {answer}\n")
            print("=" * 70)

            if "Error:" in answer:
                print("Stopping due to error. Check oMLX configuration.")
                break

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
