"""
streamlit_app.py

Bonus web UI for the Document Q&A RAG bot, built with Streamlit.

This is a thin presentation layer over the same ingestion and query
logic used by the CLI (src/ingest.py, src/query.py) -- no business
logic is duplicated here. Running this file gives a simple, deployable
web interface as an alternative to the command-line app (src/main.py).

Run locally from the project root:
    streamlit run streamlit_app.py

Deploy for free via Streamlit Community Cloud (streamlit.io/cloud) by
connecting this GitHub repo and adding GEMINI_API_KEY as a secret in
the app settings.
"""

from __future__ import annotations

import streamlit as st
from google import genai

from src import config
from src.ingest import run_ingestion
from src.query import query_rag_pipeline, load_collection


# ==========================================================================
# Page setup
# ==========================================================================

st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Document Q&A Bot")
st.caption("RAG-powered, grounded in your files — built on Gemini and ChromaDB")


# ==========================================================================
# API key handling
# ==========================================================================
# Supports both local development (.env via python-dotenv, already loaded
# inside src/config.py's import chain) and Streamlit Community Cloud
# deployment (st.secrets, configured in the app's dashboard settings).

def get_client() -> genai.Client | None:
    """
    Builds a google-genai Client using whichever API key source is
    available: Streamlit secrets first (for deployed apps), falling
    back to the local .env-loaded environment variable (for local dev).

    Returns:
        An initialized Client, or None if no API key could be found
        anywhere (the caller is responsible for showing an error).
    """
    api_key = None

    # Streamlit Cloud: secrets configured in the app dashboard.
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    # Local development fallback: .env file via config.get_api_key().
    if not api_key:
        try:
            api_key = config.get_api_key()
        except EnvironmentError:
            return None

    return genai.Client(api_key=api_key)


client = get_client()

if client is None:
    st.error(
        "No Gemini API key found. If running locally, make sure `.env` "
        "contains `GEMINI_API_KEY=...`. If deployed on Streamlit Cloud, "
        "add `GEMINI_API_KEY` under the app's Secrets settings."
    )
    st.stop()


# ==========================================================================
# Sidebar: ingestion controls
# ==========================================================================

with st.sidebar:
    st.header("📂 Document Database")

    st.markdown(
        "Documents are read from the `data/` folder in this repository. "
        "Click below to (re)build the vector database from those files."
    )

    if st.button("🔄 Ingest documents", use_container_width=True):
        with st.spinner("Reading, chunking, and embedding documents... this may take a moment."):
            try:
                run_ingestion()
                st.success("Ingestion complete!")
            except SystemExit:
                st.error(
                    "Ingestion did not complete. Check that `data/` "
                    "contains supported files (.pdf, .docx, .txt)."
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()

    # Show current database status so the user isn't guessing.
    try:
        collection = load_collection(client)
        doc_count = collection.count()
        if doc_count > 0:
            st.success(f"Database ready — {doc_count} chunk(s) indexed.")
        else:
            st.warning("Database exists but is empty. Click 'Ingest documents' above.")
    except RuntimeError:
        st.warning("No database found yet. Click 'Ingest documents' above.")

    st.divider()
    st.caption(
        "Built for the BookExpert AI Engineering Internship assignment. "
        "[View source on GitHub](https://github.com/sadukesavakalyan/document-qa-bot)"
    )


# ==========================================================================
# Main panel: chat-style question and answer
# ==========================================================================

# Session state holds chat history across reruns (Streamlit reruns the
# whole script on every interaction, so anything we want to persist --
# like prior Q&A turns -- must live in st.session_state).
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question", "answer", "citations"}

# Render prior turns first, so new ones appear at the bottom like a chat.
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["citations"]:
            with st.expander("📑 Sources"):
                for c in turn["citations"]:
                    st.markdown(f"- {c}")

# New question input, pinned to the bottom like a chat app.
question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = query_rag_pipeline(question, client=client)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        st.write(result.answer)
        if result.citations:
            with st.expander("📑 Sources"):
                for c in result.citations:
                    st.markdown(f"- {c}")

    st.session_state.history.append({
        "question": question,
        "answer": result.answer,
        "citations": result.citations,
    })
