"""The oMLX twin of rag_alice_in_wonderland.py: the same reusable in-memory Document/RAG structure, chunker,
prompts and questions (thinking still off), but the embeddings and answers come from oMLX's
OpenAI-compatible API instead of Ollama. Still raw `requests`; the chunks are embedded in batches.

Needs oMLX running on :8000 with mlx-community/embeddinggemma-300m-bf16 and
mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and OMLX_API_KEY in the environment or the
repo-root .env. Needs `pip install numpy requests python-dotenv`.
"""
from typing import List, Tuple
import os
import numpy as np
import requests
from dataclasses import dataclass
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
HEADERS = {"Authorization": f"Bearer {api_key}"}

EMBEDDING_MODEL = "mlx-community--embeddinggemma-300m-bf16"   # oMLX names Hugging Face cache models "org--repo"
LLM_MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"
BATCH_SIZE = 32   # chunks per /embeddings request
HERE = Path(__file__).parent


@dataclass
class Document:
    text: str
    chunk_id: int
    embedding: np.ndarray = None


class OMLXClient:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/v1"

    def ensure_models(self, models: List[str]) -> None:
        """Check up front that oMLX serves the required models. oMLX doesn't pull
        models the way Ollama does: it serves what is already in the Hugging Face
        cache, so a missing model gets a download hint instead of a pull."""
        try:
            response = requests.get(f"{self.base_url}/models", headers=HEADERS, timeout=10)
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not reach oMLX at {self.base_url} - is it running? "
                "Start it with 'omlx serve'."
            ) from e
        response.raise_for_status()

        installed = {m["id"] for m in response.json().get("data", [])}

        for model in models:
            if model in installed:
                continue
            repo = model.replace("--", "/", 1)
            raise RuntimeError(
                f"oMLX doesn't serve '{model}' - download it with 'hf download {repo}', "
                "then Reload models in the oMLX admin page (or 'omlx restart')."
            )

    def get_embeddings(self, texts: List[str], task_type: str = "document") -> List[np.ndarray]:
        # EmbeddingGemma uses task-specific prompts for document vs query
        if task_type == "query":
            formatted_texts = [f"task: search result | query: {text}" for text in texts]
        else:
            formatted_texts = [f"title: none | text: {text}" for text in texts]

        data = {"model": EMBEDDING_MODEL, "input": formatted_texts}
        # generous timeout: the first call also loads the model into memory
        response = requests.post(f"{self.base_url}/embeddings", headers=HEADERS, json=data, timeout=120)
        response.raise_for_status()
        return [np.array(item["embedding"]) for item in response.json()["data"]]

    def get_embedding(self, text: str, task_type: str = "document") -> np.ndarray:
        return self.get_embeddings([text], task_type)[0]

    def generate_response(self, prompt: str) -> str:
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.05,
            "top_p": 0.85,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        response = requests.post(f"{self.base_url}/chat/completions", headers=HEADERS, json=data, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class TextChunker:
    def __init__(self, chunk_size: int = 250, overlap_percentage: float = 0.2, min_chunk_ratio: float = 0.4, min_break_ratio: float = 0.75):
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.min_chunk_size = int(chunk_size * min_chunk_ratio)
        self.min_break_size = int(chunk_size * min_break_ratio)
        self.final_chunk_min_size = int(chunk_size * min_chunk_ratio)
        self.overlap_words = int(chunk_size * overlap_percentage)

    def get_config_info(self) -> str:
        return (f"Chunking: {self.chunk_size} words, {self.overlap_percentage:.0%} overlap")

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences

    def chunk_text(self, text: str) -> List[Document]:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text).strip()

        sentences = self._split_sentences(text)

        chunks = []
        chunk_id = 0
        current_chunk = []
        current_words = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())

            if (current_words + sentence_words > self.chunk_size and
                current_words >= self.min_chunk_size):

                best_break = len(current_chunk)
                for j in range(len(current_chunk) - 1, max(0, len(current_chunk) - 5), -1):
                    if ('\n' in current_chunk[j] or
                        current_chunk[j].strip().endswith(('!', '?', '."', '.\'')) and
                        len(' '.join(current_chunk[:j+1]).split()) >= self.min_break_size):
                        best_break = j + 1
                        break

                final_chunk = current_chunk[:best_break]
                chunk_text = ' '.join(final_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
                chunk_id += 1

                target_overlap_words = self.overlap_words
                overlap_sentences = []
                overlap_words = 0

                for j in range(best_break - 1, -1, -1):
                    sentence_words = len(current_chunk[j].split())
                    if overlap_words + sentence_words <= target_overlap_words:
                        overlap_sentences.insert(0, current_chunk[j])
                        overlap_words += sentence_words
                    else:
                        break

                if not overlap_sentences and best_break > 0:
                    overlap_sentences = [current_chunk[best_break - 1]]

                current_chunk = overlap_sentences
                current_words = overlap_words

            current_chunk.append(sentence)
            current_words += sentence_words
            i += 1

        if current_chunk:
            if current_words >= self.final_chunk_min_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
            elif chunks:
                chunks[-1].text += ' ' + ' '.join(current_chunk)
            else:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))

        return chunks


class GenericRAG:
    def __init__(self, file_path: str, chunker: TextChunker = None):
        self.file_path = Path(file_path)
        self.chunker = chunker if chunker else TextChunker()
        self.omlx = OMLXClient()
        self.documents: List[Document] = []
        self._load_document()

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
        self.omlx.ensure_models([EMBEDDING_MODEL, LLM_MODEL])
        print(f"Loading {self.file_path.name} - {self.chunker.get_config_info()}")

        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        text = self._clean_text(text)
        chunks = self.chunker.chunk_text(text)

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            try:
                for chunk, embedding in zip(batch, self.omlx.get_embeddings([c.text for c in batch], "document")):
                    chunk.embedding = embedding
                    self.documents.append(chunk)
                print(f"  Embedded {start + len(batch)}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Failed to embed chunks {start}-{start + len(batch) - 1}: {e}")

        print(f"Ready! {len(self.documents)} chunks loaded.")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _retrieve_chunks(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        query_embedding = self.omlx.get_embedding(query, "query")

        similarities = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _show_match_details(self, matches: List[Tuple[Document, float]], query: str) -> None:
        if not matches:
            return

        query_embedding = self.omlx.get_embedding(query, "query")
        all_scores = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            all_scores.append(score)

        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        std_dev = variance ** 0.5 if variance > 0 else 0.001

        print("\nBest matches (cosine similarity scores & significance):")
        for i, (doc, score) in enumerate(matches, 1):
            significance = (score - mean_score) / std_dev if std_dev > 0 else 0
            preview = doc.text[:80].replace('\n', ' ') + "..."
            print(f"  {i}. Chunk: {doc.chunk_id:03d} | Score: {score:.4f} | "
                  f"Significance: {significance:+.2f}σ | \"{preview}\"")
        print()

    def query(self, question: str, show_matches: bool = False) -> str:
        if not question.strip():
            return "Please provide a question."

        relevant_docs = self._retrieve_chunks(question)

        if not relevant_docs:
            return "I couldn't find relevant information to answer your question."

        if show_matches:
            self._show_match_details(relevant_docs, question)

        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            context_parts.append(f"Context {i}:\n{doc.text}")

        context = "\n\n".join(context_parts)

        prompt = f"""Answer the question using the provided context passages.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the context doesn't contain enough information, say so clearly.

Question: {question}

Context:
{context}

Answer:"""

        try:
            answer = self.omlx.generate_response(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    print("Generic RAG Demo System")
    print("Make sure oMLX is running on port 8000\n")

    try:
        rag = GenericRAG(str(HERE / "data" / "alice_in_wonderland.txt"))
        print()

        questions = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "Where was the Cheshire Cat when Alice first met him?",
            "What happens when Alice falls down the rabbit hole?",
            "What did the White Rabbit say when Alice first saw him?",
            "Was stand auf der Flasche, die Alice schrumpfen ließ?",
            "Que faisait Alice au début de l'histoire?",
            "¿Qué dijo el Conejo Blanco cuando Alice lo vio por primera vez?"
        ]

        for question in questions:
            print("=" * 70)
            print(f"Q: {question}")
            print("=" * 70)

            answer = rag.query(question, show_matches=True)
            print(f"\nA: {answer}\n")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
