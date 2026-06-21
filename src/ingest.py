"""
ingest.py

Document ingestion pipeline for the RAG Q&A bot.

Responsibilities:
    1. Scan the data/ folder for supported files (PDF, DOCX, TXT)
    2. Extract text from each file, preserving source/page metadata
    3. Split extracted text into overlapping chunks
    4. Embed each chunk using Gemini's gemini-embedding-001 model
    5. Persist chunks + embeddings + metadata into a local ChromaDB collection

Run standalone from the project root:
    python -m src.ingest

This is intentionally separate from query.py (see query.py) so that
indexing (slow, costly, run rarely) is decoupled from querying
(fast, run often).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument
from tqdm import tqdm

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types

from src import config

# Load GEMINI_API_KEY (and any other vars) from .env into the environment.
load_dotenv()


# ==========================================================================
# Step 1: Document extraction
# ==========================================================================

def extract_pdf(file_path: Path) -> list[dict]:
    """
    Extracts text page-by-page from a PDF file.

    Args:
        file_path: Path to the .pdf file.

    Returns:
        A list of dicts, one per non-empty page, each shaped as:
        {"text": str, "metadata": {"source": str, "page": int}}
        Returns an empty list (with a printed warning) if the file
        cannot be read or parsed.
    """
    extracted: list[dict] = []
    file_name = file_path.name

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        print(f"  [WARN] Could not open PDF '{file_name}': {e}")
        return extracted

    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"  [WARN] Could not extract page {index + 1} of '{file_name}': {e}")
            continue

        if text and text.strip():
            clean_text = " ".join(text.split())
            extracted.append({
                "text": clean_text,
                "metadata": {
                    "source": file_name,
                    "page": index + 1,  # 1-indexed for human readability
                },
            })

    if not extracted:
        print(f"  [WARN] No extractable text found in '{file_name}'. "
              f"It may be a scanned/image-only PDF (pypdf cannot OCR).")

    return extracted


def extract_docx(file_path: Path) -> list[dict]:
    """
    Extracts text from a DOCX file.

    DOCX files have no native concept of a fixed "page" the way PDFs do
    (page breaks depend on the rendering app, font, screen size, etc.),
    so true page numbers are not available here. Instead, we group
    paragraphs into pseudo-pages of a fixed paragraph count, purely to
    give citations *some* sub-document locator that's more useful than
    "somewhere in this file." This is documented clearly in the README
    as a known limitation, per the assignment's own guidance.

    Args:
        file_path: Path to the .docx file.

    Returns:
        A list of dicts shaped as:
        {"text": str, "metadata": {"source": str, "page": int}}
        where "page" here means "paragraph block number", not a true
        page. Returns an empty list (with a printed warning) on failure.
    """
    extracted: list[dict] = []
    file_name = file_path.name
    PARAGRAPHS_PER_BLOCK = 10  # arbitrary grouping for citation granularity

    try:
        doc = DocxDocument(str(file_path))
    except Exception as e:
        print(f"  [WARN] Could not open DOCX '{file_name}': {e}")
        return extracted

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]

    if not paragraphs:
        print(f"  [WARN] No extractable text found in '{file_name}'.")
        return extracted

    for block_index in range(0, len(paragraphs), PARAGRAPHS_PER_BLOCK):
        block = paragraphs[block_index:block_index + PARAGRAPHS_PER_BLOCK]
        block_text = " ".join(block)
        clean_text = " ".join(block_text.split())
        if clean_text:
            extracted.append({
                "text": clean_text,
                "metadata": {
                    "source": file_name,
                    # "page" is really "paragraph block #" for DOCX — see docstring.
                    "page": (block_index // PARAGRAPHS_PER_BLOCK) + 1,
                },
            })

    return extracted


def extract_txt(file_path: Path) -> list[dict]:
    """
    Extracts text from a plain .txt file.

    TXT files have no page concept at all, so metadata["page"] is fixed
    at 1 for the whole file, as explicitly permitted by the assignment
    spec for files without page-level granularity.

    Args:
        file_path: Path to the .txt file.

    Returns:
        A list containing zero or one dict shaped as:
        {"text": str, "metadata": {"source": str, "page": 1}}
    """
    file_name = file_path.name

    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] Could not read TXT '{file_name}': {e}")
        return []

    clean_text = " ".join(raw_text.split())
    if not clean_text:
        print(f"  [WARN] '{file_name}' is empty.")
        return []

    return [{
        "text": clean_text,
        "metadata": {
            "source": file_name,
            "page": 1,
        },
    }]


def extract_file(file_path: Path) -> list[dict]:
    """
    Dispatches extraction to the correct handler based on file extension.

    Args:
        file_path: Path to a supported file (.pdf, .docx, .txt).

    Returns:
        A list of page/block-level text dicts (see individual extractors).
        Returns an empty list for unsupported extensions.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(file_path)
    elif suffix == ".docx":
        return extract_docx(file_path)
    elif suffix == ".txt":
        return extract_txt(file_path)
    else:
        print(f"  [WARN] Skipping unsupported file type: {file_path.name}")
        return []


# ==========================================================================
# Step 2: Recursive / paragraph-aware text chunking
# ==========================================================================

def _split_on_separator(text: str, separator: str) -> list[str]:
    """Splits text on a separator, keeping the separator out of pieces."""
    if not separator:
        return list(text)
    return [piece for piece in text.split(separator) if piece]


def recursive_split(text: str, chunk_size: int) -> list[str]:
    """
    Recursively splits text into pieces no longer than chunk_size,
    preferring to break on paragraph boundaries, then line breaks,
    then spaces, and only falling back to a hard character cut as
    a last resort. This avoids slicing a chunk mid-word or mid-sentence
    whenever a cleaner boundary is available nearby.

    Args:
        text: The text to split.
        chunk_size: Maximum length (in characters) of each returned piece.

    Returns:
        A list of text pieces, each <= chunk_size characters (best-effort;
        a single "word" longer than chunk_size will still be hard-cut).
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", " "]

    for sep in separators:
        pieces = _split_on_separator(text, sep)
        if len(pieces) <= 1:
            continue  # this separator didn't help, try the next one

        # Greedily pack pieces into lines up to chunk_size.
        merged: list[str] = []
        current = ""
        for piece in pieces:
            candidate = (current + sep + piece) if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # If a single piece is itself too big, recurse on it
                # with the next-finer separator.
                if len(piece) > chunk_size:
                    merged.extend(recursive_split(piece, chunk_size))
                    current = ""
                else:
                    current = piece
        if current:
            merged.append(current)

        if merged:
            return merged

    # Last resort: no separator helped (e.g. one giant unbroken token) —
    # hard-cut at chunk_size.
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_extracted_pages(
    pages: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """
    Splits page/block-level extracted text into overlapping chunks,
    carrying forward each chunk's source metadata.

    Uses a recursive, paragraph-aware splitter (see recursive_split)
    rather than a naive fixed-width slice, so chunks tend to break on
    natural boundaries (paragraphs, lines, words) instead of mid-sentence.
    Overlap is then applied by sliding a window with character-level
    carry-over between consecutive pieces, preserving the original
    fixed-overlap behavior the assignment specifies.

    Args:
        pages: Output of extract_file() — list of {"text", "metadata"} dicts.
        chunk_size: Target max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks
            within the same page/block.

    Returns:
        A list of dicts: {"text": str, "metadata": {..., "chunk_index": int}}
    """
    all_chunks: list[dict] = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        if not text:
            continue

        # First pass: clean recursive split into target-size pieces.
        pieces = recursive_split(text, chunk_size)

        # Second pass: stitch in overlap by prepending the tail of the
        # previous piece onto the next one, so context isn't lost at
        # piece boundaries.
        stitched: list[str] = []
        for i, piece in enumerate(pieces):
            if i == 0 or chunk_overlap <= 0:
                stitched.append(piece)
            else:
                prev = pieces[i - 1]
                overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                stitched.append((overlap_text + " " + piece).strip())

        for chunk_index, chunk_text in enumerate(stitched):
            if not chunk_text.strip():
                continue
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                },
            })

    return all_chunks


# ==========================================================================
# Step 3: Gemini embedding function (ChromaDB-compatible)
# ==========================================================================

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    A ChromaDB-compatible embedding function backed by Google's
    gemini-embedding-001 model, via the current `google-genai` SDK.

    NOTE: ChromaDB ships a built-in Google embedding function, but it is
    written against the deprecated `google-generativeai` package. This
    project defines its own thin wrapper around the current `google-genai`
    client so it isn't tied to a legacy SDK that Google is sunsetting.

    Uses task_type="RETRIEVAL_DOCUMENT" since this embedding function is
    used for *storing* chunks during ingestion. query.py uses a separate
    "RETRIEVAL_QUERY" task type when embedding the user's question —
    Gemini's embedding model is trained to produce better-aligned vectors
    for retrieval when each side of the document/query pair is tagged
    correctly.
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str = config.EMBEDDING_MODEL,
        output_dimensionality: int = config.EMBEDDING_OUTPUT_DIMENSIONALITY,
    ):
        self._client = client
        self._model_name = model_name
        self._output_dimensionality = output_dimensionality

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds a batch of text documents for storage (RETRIEVAL_DOCUMENT).

        Args:
            input: List of strings to embed.

        Returns:
            List of embedding vectors (list[float]), one per input string.
        """
        result = self._client.models.embed_content(
            model=self._model_name,
            contents=input,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._output_dimensionality,
            ),
        )
        return [embedding.values for embedding in result.embeddings]

    def name(self) -> str:
        return "gemini-embedding-001"


# ==========================================================================
# Step 4: Deterministic chunk IDs (prevents duplicate entries on re-ingest)
# ==========================================================================

def make_chunk_id(metadata: dict) -> str:
    """
    Builds a deterministic, stable ID for a chunk based on its source
    file, page/block number, and chunk index.

    Using a deterministic ID (instead of a random UUID or row counter)
    means re-running ingestion on unchanged files calls collection.upsert()
    with the *same* IDs, so chunks are updated in place rather than
    duplicated. This directly satisfies the assignment requirement that
    "repeated ingestion does not create duplicate entries."

    Args:
        metadata: A chunk's metadata dict, must contain "source",
            "page", and "chunk_index".

    Returns:
        A short, stable hex string ID.
    """
    raw_key = f"{metadata['source']}::{metadata['page']}::{metadata['chunk_index']}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]


# ==========================================================================
# Step 5: Orchestration — scan, extract, chunk, embed, persist
# ==========================================================================

def scan_data_dir(data_dir: Path) -> list[Path]:
    """
    Scans the data directory for files with supported extensions.

    Args:
        data_dir: Path to the data/ folder.

    Returns:
        Sorted list of file paths to ingest. Empty list if the folder
        doesn't exist or contains no supported files.
    """
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return []

    files = [
        f for f in sorted(data_dir.iterdir())
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]
    return files


def run_ingestion(data_dir: Path | None = None, db_dir: Path | None = None) -> None:
    """
    Runs the full ingestion pipeline end to end:
    scan -> extract -> chunk -> embed -> persist.

    Args:
        data_dir: Override for the data folder (defaults to config.DATA_DIR).
        db_dir: Override for the ChromaDB persistence folder
            (defaults to config.DB_DIR).
    """
    data_dir = data_dir or config.DATA_DIR
    db_dir = db_dir or config.DB_DIR

    print(f"Scanning '{data_dir}' for supported files {sorted(config.SUPPORTED_EXTENSIONS)}...")
    files = scan_data_dir(data_dir)

    if not files:
        print("[ERROR] No supported documents found. "
              f"Add .pdf, .docx, or .txt files to '{data_dir}' and re-run.")
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  - {f.name}")

    # --- Extraction ---
    print("\nExtracting text...")
    all_pages: list[dict] = []
    for file_path in tqdm(files, desc="Extracting", unit="file"):
        pages = extract_file(file_path)
        all_pages.extend(pages)

    if not all_pages:
        print("[ERROR] No extractable text found in any file. Nothing to ingest.")
        sys.exit(1)

    print(f"Extracted {len(all_pages)} page/block segment(s) of text.")

    # --- Chunking ---
    print("\nChunking text...")
    chunks = chunk_extracted_pages(all_pages)
    print(f"Produced {len(chunks)} chunk(s) "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    # --- API key + client setup ---
    try:
        api_key = config.get_api_key()
    except EnvironmentError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    embedding_fn = GeminiEmbeddingFunction(client=client)

    # --- Persist to ChromaDB ---
    print(f"\nConnecting to ChromaDB at '{db_dir}'...")
    db_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(db_dir))

    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [make_chunk_id(chunk["metadata"]) for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print(f"Embedding and upserting {len(chunks)} chunk(s) into ChromaDB "
          f"(this calls the Gemini embeddings API and may take a while)...")

    # Batch in groups to avoid oversized single API calls and to show progress.
    BATCH_SIZE = 50
    try:
        for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding+Upserting", unit="batch"):
            end = start + BATCH_SIZE
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
    except Exception as e:
        print(f"\n[ERROR] Failed during embedding/upsert: {e}")
        print("Common causes: invalid/missing GEMINI_API_KEY, network issue, "
              "or API quota exceeded.")
        sys.exit(1)

    # upsert() means re-running ingestion on the same files updates chunks
    # in place rather than duplicating them, since IDs are deterministic.
    print(f"\nSuccessfully indexed {len(chunks)} chunk(s) into "
          f"collection '{config.COLLECTION_NAME}' at '{db_dir}'.")
    print(f"Total documents now in collection: {collection.count()}")


if __name__ == "__main__":
    run_ingestion()
