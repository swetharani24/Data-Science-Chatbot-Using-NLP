"""Measure top-1 / top-3 accuracy on paraphrased queries and out-of-scope rejection."""
import sys
from chatbot import DSChatbot

TEST = [
    ("my model does great on train but badly on test", "overfitting"),
    ("how to deal with rows that have NaN", "missing_values"),
    ("explain k fold validation", "cross_validation"),
    ("difference between precision and recall", "precision_recall"),
    ("what does PCA do", "pca"),
    ("lasso vs ridge", "regularization"),
    ("how do trees in a forest vote", "random_forest"),
    ("xgboost explained", "boosting"),
    ("how to pick number of clusters", "kmeans"),
    ("what is statistical significance p value", "hypothesis_test"),
    ("how do I convert text to vectors with word importance", "tfidf"),
    ("self attention architecture", "transformers"),
    ("class imbalance fraud detection", "imbalanced"),
    ("test info leaking into training", "data_leakage"),
    ("steps in a data science project", "pipeline"),
]
OUT_OF_SCOPE = ["what's the weather today", "who won the cricket match", "recipe for biryani"]

backend = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
bot = DSChatbot(backend=backend)

top1 = top3 = 0
for q, gold in TEST:
    m = [x["id"] for x in bot.respond(q)["matches"]]
    top1 += m[0] == gold
    top3 += gold in m
    if m[0] != gold:
        print(f"MISS: {q!r} -> {m[0]} (expected {gold})")

rejected = sum(bot.respond(q)["answer"].startswith("I'm not sure") for q in OUT_OF_SCOPE)
print(f"[{backend}] top-1: {top1/len(TEST):.0%} | top-3: {top3/len(TEST):.0%} | "
      f"out-of-scope rejected: {rejected}/{len(OUT_OF_SCOPE)}")
