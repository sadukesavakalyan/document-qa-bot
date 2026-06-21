"""
config.py

Centralizes all configuration constants for the Document Q&A RAG bot:
file paths, chunking parameters, model names, and supported file types.

Keeping these values in one place means ingest.py and query.py never
hardcode "magic" values, and changing a setting (e.g. chunk size, model
name) only requires editing this single file.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Path configuration
# --------------------------------------------------------------------------

# Project root = the directory that contains src/, data/, db/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Folder the user drops their source documents into (PDF, DOCX, TXT)
DATA_DIR: Path = PROJECT_ROOT / "data"

# Folder where ChromaDB persists its on-disk vector index
DB_DIR: Path = PROJECT_ROOT / "db"

# Name of the ChromaDB collection that stores all document chunks
COLLECTION_NAME: str = "document_knowledge_base"

# --------------------------------------------------------------------------
# Supported file types
# --------------------------------------------------------------------------

# Lowercase file extensions the ingestion pipeline knows how to parse.
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}

# --------------------------------------------------------------------------
# Chunking configuration
# --------------------------------------------------------------------------

# Target size (in characters) for each text chunk.
# ~1000 chars is a reasonable middle ground: small enough to keep retrieval
# precise, large enough to preserve sentence/paragraph context.
CHUNK_SIZE: int = 1000

# Number of overlapping characters between consecutive chunks.
# Protects against losing meaning when a key sentence falls on a cut boundary.
CHUNK_OVERLAP: int = 200

# --------------------------------------------------------------------------
# Retrieval configuration
# --------------------------------------------------------------------------

# Default number of top-k chunks to retrieve per query.
TOP_K: int = 4

# --------------------------------------------------------------------------
# Model configuration (Google Gemini via the google-genai SDK)
# --------------------------------------------------------------------------
#
# NOTE on SDK choice:
# Google has deprecated the older `google-generativeai` package in favor of
# the unified `google-genai` SDK (client-based pattern: `genai.Client()`).
# This project targets the current, supported SDK. If you started this
# project from older tutorials using `import google.generativeai as genai`,
# that pattern is no longer recommended by Google and may stop working
# entirely as the legacy package is sunset.
#
# NOTE on EMBEDDING_MODEL — deviates from the original assignment spec:
# The assignment doc specifies "text-embedding-004", but Google deprecated
# that model on January 14, 2026 (confirmed via 404 NOT_FOUND when calling
# it live during this project's development). The current replacement is
# "gemini-embedding-001". This is a *required* substitution, not a style
# choice — the old model name no longer resolves at all.
#
# gemini-embedding-001 defaults to 3072-dimensional vectors. We truncate to
# 768 via output_dimensionality for a smaller/faster local ChromaDB index,
# which Google's docs confirm loses negligible quality (Matryoshka
# Representation Learning). Must stay consistent between ingest.py (storing
# vectors) and query.py (embedding the user's question) or similarity
# search breaks.

# NOTE on GENERATION_MODEL — deviates from the original assignment spec:
# The assignment doc specifies "gemini-2.5-flash-preview-09-2025", but this
# dated preview model returned 404 NOT_FOUND when called live during this
# project's development (confirmed against Google's own deprecation
# announcements: that exact preview build is being discontinued). Google's
# guidance for anyone hitting this 404 is explicit: switch to the
# undated "gemini-2.5-flash" alias.
#
# IMPORTANT: Gemini model names are being retired on a rolling basis
# (multiple times per year). If this model 404s again by the time you
# read this, check https://ai.google.dev/gemini-api/docs/models for
# the current recommended Flash-tier model and swap the string below —
# the rest of the pipeline (prompt, retrieval, grounding) does not need
# to change, only this one constant.
GENERATION_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIMENSIONALITY: int = 768

# Name of the environment variable holding the Gemini API key.
API_KEY_ENV_VAR: str = "GEMINI_API_KEY"

# --------------------------------------------------------------------------
# Grounding / hallucination-control configuration
# --------------------------------------------------------------------------

# Exact refusal message the LLM must use when the answer isn't in context.
# Defined centrally so query.py, main.py, and tests all reference the same
# string instead of duplicating it.
NO_ANSWER_MESSAGE: str = (
    "I am sorry, but the provided documents do not contain the answer "
    "to your question."
)


def get_api_key() -> str:
    """
    Reads the Gemini API key from the environment and validates it exists.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the API key is missing or empty, with a
            clear message telling the user how to fix it.
    """
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key or not api_key.strip():
        raise EnvironmentError(
            f"Missing required environment variable: {API_KEY_ENV_VAR}.\n"
            f"Fix: copy .env.example to .env and set {API_KEY_ENV_VAR}=<your key>.\n"
            f"Get a key at: https://aistudio.google.com/app/apikey"
        )
    return api_key.strip()
