"""The oMLX twin of embedding_example.py: the same interactive sentence matcher with mean, standard deviation
and z-scores, but the EmbeddingGemma vectors come from oMLX's OpenAI-compatible REST endpoint
(/v1/embeddings, still raw `requests`) instead of Ollama's.

Needs oMLX running on :8000 with mlx-community/embeddinggemma-300m-bf16 in the Hugging Face cache, and
OMLX_API_KEY in the environment or the repo-root .env. Needs `pip install numpy requests python-dotenv`.
"""
import os

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")


def get_omlx_embeddings(texts: list) -> list:
    # one request embeds every text; oMLX names Hugging Face cache models "org--repo"
    data = {"model": "mlx-community--embeddinggemma-300m-bf16", "input": texts}
    try:
        response = requests.post("http://127.0.0.1:8000/v1/embeddings", json=data,
                                 headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar_sentence(question: str, sentences: list) -> tuple:
    embeddings = get_omlx_embeddings([question] + sentences)
    if embeddings is None:
        return [], 0, 0
    question_embedding, sentence_embeddings = embeddings[0], embeddings[1:]

    similarities = []
    similarity_scores = []
    for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
        similarity = cosine_similarity(question_embedding, sentence_embedding)
        similarities.append((sentence, similarity))
        similarity_scores.append(similarity)

    if not similarity_scores:
        return [], 0, 0

    mean_similarity = np.mean(similarity_scores)
    std_similarity = np.std(similarity_scores)

    # z-score: standard deviations from the mean, so the caller can flag outliers
    enhanced_similarities = [
        (s, sim, (sim - mean_similarity) / std_similarity if std_similarity > 0 else 0)
        for s, sim in similarities
    ]
    enhanced_similarities.sort(key=lambda x: x[1], reverse=True)
    return enhanced_similarities, mean_similarity, std_similarity


def main():
    sentences = [
        "The cat sat on the mat.",
        "What is the weather like today?",
        "Python is a popular programming language.",
        "How do I write code?",
        "The quick brown fox jumps over the lazy dog.",
        "What's the best way to learn programming?",
        "Machine learning involves training models on data.",
        "Can you help me debug this code?",
    ]

    while True:
        question = input("\nEnter your question (or 'quit' / Enter to exit): ").strip()
        if not question or question.lower() == 'quit':   # an empty question has no embedding
            break

        print("\nFinding similar sentences...")
        results, mean_sim, std_sim = find_most_similar_sentence(question, sentences)

        if results:
            print(f"\nStatistics:")
            print(f"Mean similarity: {mean_sim:.4f}")
            print(f"Standard deviation: {std_sim:.4f}")
            print("\nResults (sorted by similarity):")
            print("=" * 70)
            for sentence, similarity, std_devs in results:
                print(f"Similarity: {similarity:.4f}, {std_devs:+.2f}σ")
                print(f"Sentence: {sentence}")
                print("-" * 70)
        else:
            print("No results found.")


if __name__ == "__main__":
    print("Enhanced Sentence Similarity Matcher")
    print("Make sure oMLX is running locally on port 8000")
    main()
