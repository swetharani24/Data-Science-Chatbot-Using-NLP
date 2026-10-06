"""Retrieval-based Data Science chatbot (TF-IDF or sentence-transformer backend)."""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GREETINGS = re.compile(r"^\s*(hi|hello|hey|good (morning|evening)|howdy)\b", re.I)
THANKS = re.compile(r"\b(thanks|thank you|thx)\b", re.I)
FALLBACK = ("I'm not sure about that one. Try asking about topics like overfitting, "
            "cross-validation, PCA, TF-IDF, random forests or A/B testing.")


def clean(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^a-z0-9\s\-]", " ", text)


class DSChatbot:
    def __init__(self, kb_path=None, backend="tfidf", threshold=None):
        if kb_path is None:
            base = Path(__file__).parent
            kb_path = base / "data" / "kb.json"
            if not kb_path.exists():  # flat download: kb.json next to chatbot.py
                kb_path = base / "kb.json"
        self.kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        self.backend = backend
        self.docs = [f"{e['question']} {e['question']} {e['answer']} {' '.join(e['tags'])}"
                     for e in self.kb]
        if backend == "embeddings":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.matrix = self.model.encode(self.docs, normalize_embeddings=True)
            self.threshold = 0.35 if threshold is None else threshold
        else:
            self.vec = TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2),
                                       stop_words="english", sublinear_tf=True)
            self.matrix = self.vec.fit_transform(self.docs)
            self.threshold = 0.12 if threshold is None else threshold

    def _scores(self, query):
        if self.backend == "embeddings":
            q = self.model.encode([query], normalize_embeddings=True)
        else:
            q = self.vec.transform([query])
        return cosine_similarity(q, self.matrix).ravel()

    def respond(self, query: str, top_k: int = 3) -> dict:
        if GREETINGS.match(query):
            return {"answer": "Hi! Ask me anything about data science and machine learning.",
                    "score": 1.0, "matches": []}
        if THANKS.search(query):
            return {"answer": "You're welcome!", "score": 1.0, "matches": []}

        scores = self._scores(query)
        order = scores.argsort()[::-1][:top_k]
        best = order[0]
        matches = [{"id": self.kb[i]["id"], "question": self.kb[i]["question"],
                    "score": float(scores[i])} for i in order]
        if scores[best] < self.threshold:
            return {"answer": FALLBACK, "score": float(scores[best]), "matches": matches}
        return {"answer": self.kb[best]["answer"], "score": float(scores[best]),
                "matches": matches}


if __name__ == "__main__":
    bot = DSChatbot()
    print("DS Bot ready. Type 'quit' to exit.")
    while (q := input("You: ").strip()).lower() not in {"quit", "exit"}:
        print("Bot:", bot.respond(q)["answer"])
