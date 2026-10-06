# Data Science Chatbot (end-to-end NLP project)

A retrieval-based chatbot that answers data science / ML questions.

## Run it
```bash
pip install -r requirements.txt
python chatbot.py                 # terminal chat
python evaluate.py tfidf          # evaluation (or: embeddings)
streamlit run app.py              # web UI
```

## Project phases
1. **Problem definition**: answer DS questions from a curated knowledge base; reject out-of-scope questions.
2. **Data collection**: `data/kb.json` (28 Q&A entries: id, question, answer, tags). Grow it from textbooks, docs, StackExchange, your own notes.
3. **Preprocessing**: lowercase, strip punctuation, stop-word removal, uni+bigrams (`clean()` in `chatbot.py`).
4. **Representation**: TF-IDF (baseline) or `all-MiniLM-L6-v2` sentence embeddings (`backend="embeddings"`).
5. **Retrieval**: cosine similarity, top-k matches; below a threshold the bot says it doesn't know.
6. **Evaluation**: `evaluate.py` reports top-1/top-3 accuracy on paraphrased queries and out-of-scope rejection.
7. **Deployment**: Streamlit app (`app.py`); containerize with Docker or deploy to Streamlit Cloud / Hugging Face Spaces.
8. **Monitoring**: log unanswered queries, add them to the KB, re-evaluate.

## Baseline result (TF-IDF)
top-1 80%, top-3 87%, out-of-scope rejection 3/3. Misses are paraphrases with no keyword overlap, which is the case embeddings fix.

## Upgrades
- Run `python evaluate.py embeddings` and compare.
- Add conversation memory to resolve follow-ups ("what about L1?").
- Intent classifier (Logistic Regression / DistilBERT) before retrieval.
- RAG: pass top-k passages to an LLM to generate answers grounded in your KB.
- Scale the KB to 500+ entries and use FAISS for search.
- Add fuzzy spell correction (`rapidfuzz`) for typos.
