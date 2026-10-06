"""Run with: streamlit run app.py"""
import streamlit as st
from chatbot import DSChatbot

st.set_page_config(page_title="DS Bot", page_icon="📊")
st.title("📊 Data Science Chatbot")

backend = st.sidebar.radio("Backend", ["tfidf", "embeddings"])
show_debug = st.sidebar.checkbox("Show matches & scores")


@st.cache_resource
def load(b):
    return DSChatbot(backend=b)


bot = load(backend)
if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

if q := st.chat_input("Ask a data science question..."):
    st.chat_message("user").write(q)
    res = bot.respond(q)
    st.chat_message("assistant").write(res["answer"])
    if show_debug:
        st.sidebar.json(res["matches"])
    st.session_state.history += [("user", q), ("assistant", res["answer"])]
