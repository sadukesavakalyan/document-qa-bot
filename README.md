# Document Q&A Bot — RAG-Powered, Grounded in Your Files

A Retrieval-Augmented Generation (RAG) application that answers questions
about your own documents (PDF, DOCX, TXT) using Google Gemini — without
hallucinating, and with inline source citations for every fact.

Built for the BookExpert AI Engineering Internship assignment.

---

## 1. What This Project Does

Large language models know a lot about the world in general, but nothing
about *your* specific documents — your resume, your company's policy PDF,
your research paper. They also tend to confidently make things up when
asked about content they don't actually know.

This project solves both problems with RAG:

1. Your documents are split into small chunks of text.
2. Each chunk is converted into a vector embedding and stored locally.
3. When you ask a question, the question is embedded the same way, and
   the most relevant chunks are retrieved by similarity search.
4. Those chunks — and *only* those chunks — are handed to Gemini along
   with your question, with strict instructions to answer using nothing
   else.
5. The answer comes back with inline citations pointing to the exact
   file and page/section the information came from.

If the answer genuinely isn't in your documents, the bot says so
explicitly instead of guessing.

---

## 2. Architecture

```
                         ┌─────────────────────┐
                         │   data/ (PDF, DOCX,  │
                         │       TXT files)     │
                         └──────────┬───────────┘
                                    │
                              ingest.py
                    (extract → chunk → embed)
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   ChromaDB (db/)     │
                         │  persistent, local   │
                         └──────────┬───────────┘
                                    │
                              query.py
                  (embed question → retrieve top-k
                       → build grounded prompt
                            → call Gemini)
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Answer + Citations  │
                         └──────────┬───────────┘
                                    │
                              main.py (CLI)
```

**Why ingestion and querying are separate scripts:** building the vector
database (embedding every chunk) costs API calls and time, and only
needs to happen when your documents change. Querying needs to be fast
and run on every single question. Keeping them as separate modules
(`ingest.py` and `query.py`) means you can re-ingest only when needed,
without re-embedding everything just to ask one question.

### File-by-file

| File | Responsibility |
|---|---|
| `src/config.py` | All constants in one place: paths, chunk size/overlap, top-k, model names |
| `src/ingest.py` | Scans `data/`, extracts text, chunks it, embeds it, saves to ChromaDB |
| `src/query.py` | Loads the database, retrieves relevant chunks, builds the grounded prompt, calls Gemini |
| `src/main.py` | Interactive CLI tying ingestion and querying together |

---

## 3. A Note on Model Names (Important — Read This)

This project intentionally **deviates from the original assignment brief**
in two specific, necessary ways. Both were discovered by actually running
the code against the live Gemini API during development, not assumed in
advance:

1. **SDK:** The assignment brief specifies `google-generativeai`. That
   package is deprecated by Google in favor of the unified `google-genai`
   SDK (the `genai.Client()` pattern). This project uses the current SDK.

2. **Embedding model:** The brief specifies `text-embedding-004`. That
   model was deprecated by Google on January 14, 2026, and returns a
   live 404 error. This project uses its replacement,
   `gemini-embedding-001`, truncated to 768 dimensions via
   `output_dimensionality` for a smaller, faster local index (Google's
   own documentation confirms this loses negligible retrieval quality).

3. **Generation model:** The brief specifies
   `gemini-2.5-flash-preview-09-2025`, a dated preview build that also
   404s — Google's own deprecation notices confirm this exact preview
   snapshot is being retired. This project uses the stable, undated
   `gemini-2.5-flash` alias instead.

**Why this matters for evaluation:** Google is currently retiring Gemini
model names on a rolling basis, faster than usual. If `gemini-2.5-flash`
also 404s by the time this is reviewed, that is expected, not a bug in
this code — check
[ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)
for the current recommended Flash-tier model and update the single
`GENERATION_MODEL` constant in `src/config.py`. No other code needs to
change. The same applies to `EMBEDDING_MODEL` if Google retires
`gemini-embedding-001` in the future.

---

## 4. Setup

### Prerequisites

- Python 3.11 or higher (this project was built and tested on 3.13;
  **avoid Python 3.9** — its bundled SQLite version is too old for
  ChromaDB and will fail with a `RuntimeError` about `sqlite3`)
- A free Gemini API key from
  [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### Windows

```cmd
git clone <your-repo-url>
cd document-qa-bot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then open `.env` in a text editor and replace the placeholder with your
real API key:

```
GEMINI_API_KEY=your_actual_key_here
```

### macOS / Linux

```bash
git clone <your-repo-url>
cd document-qa-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` the same way as above.

---

## 5. Usage

### Step 1 — Add your documents

Drop PDF, DOCX, and/or TXT files into the `data/` folder. The repo ships
with a few sample files already there.

### Step 2 — Run the interactive CLI

```bash
python -m src.main
```

You'll see a menu:

```
[1] Ingest documents (scan data/ and (re)build the database)
[2] Ask a question
[3] Exit
```

Choose **1** first to build the vector database (this calls the Gemini
embeddings API once per chunk — expect a short delay). Then choose **2**
to ask questions repeatedly. Each answer can optionally show its source
citations. Type `back` to return to the menu, or `3` to exit.

### Running ingestion or querying directly (without the menu)

```bash
# Build/rebuild the database
python -m src.ingest

# Ask a single one-off question
python -m src.query "What is the candidate's email address?"
```

Re-running `ingest.py` on the same files does **not** create duplicate
entries — each chunk gets a deterministic ID derived from its source
file, page, and position, so re-ingestion updates existing entries in
place rather than duplicating them.

---

## 6. How Hallucination Control Works

Every prompt sent to Gemini includes an explicit system instruction:

> Answer the user's question using ONLY the context provided below —
> never use your own outside knowledge, even if you happen to know the
> answer... If the answer cannot be found in the provided context,
> respond with EXACTLY this sentence and nothing else: "I am sorry, but
> the provided documents do not contain the answer to your question."

This is enforced at the prompt level (instructing the model not to use
outside knowledge) rather than at the retrieval level (e.g. a similarity
score cutoff), because a hard score threshold is brittle — it varies by
embedding model and document set. Combining retrieval with an explicit
grounding instruction is the standard, more robust approach.

If the database is empty, or no chunks are retrieved at all, the
application returns the refusal message immediately **without even
calling the generation model** — saving an API call and guaranteeing
the answer is never an outside-knowledge guess.

---

## 7. How Citations Work

Every chunk carries metadata about exactly where it came from:

- **PDF:** real, accurate page numbers (1-indexed)
- **DOCX:** see limitation below — paragraph-block numbers, not true pages
- **TXT:** fixed at page 1 (no page concept in plain text)

When chunks are retrieved, this metadata is formatted into citation
labels like `Source: report.pdf, Page: 4` and injected directly into the
prompt alongside the chunk text, with an explicit instruction for Gemini
to cite them inline next to every fact. The same metadata is also
returned separately as a structured list, so the CLI can display
"Sources" independently of the answer text.

---

## 8. Known Limitations

- **DOCX page numbers aren't real page numbers.** Word documents have no
  fixed concept of a "page" the way PDFs do — page breaks depend on the
  viewing application, font, and screen size. This project groups DOCX
  paragraphs into blocks of 10 and labels each block a "page" purely to
  give citations *some* sub-document locator more useful than "somewhere
  in this file." This is a deliberate, documented simplification, not a
  bug.
- **Scanned/image-only PDFs won't extract any text.** `pypdf` reads
  embedded text layers; it cannot OCR a scanned image. Such files will
  print a warning during ingestion and contribute no chunks.
- **No reranking.** Retrieval is a single vector similarity search.
  Results are not re-scored by a separate, more precise reranking model.
- **No score-threshold filtering.** All top-k retrieved chunks are
  passed to the model regardless of how relevant they actually are;
  there's no minimum similarity cutoff to exclude weak matches.
- **Single embedding/generation provider.** The pipeline is built
  specifically around Gemini's APIs and would need adapter changes to
  support another provider (e.g. OpenAI).

---

## 9. Future Improvements

- **Reranking:** add a cross-encoder reranking step after initial
  retrieval to improve precision on borderline matches.
- **Score thresholds:** drop retrieved chunks below a minimum similarity
  score instead of always returning a fixed top-k.
- **Metadata filtering:** let users restrict a query to a specific
  source file or page range.
- **True DOCX pagination:** render DOCX to PDF first (e.g. via
  LibreOffice headless) to get accurate page numbers instead of
  paragraph blocks.
- **Streamlit UI:** a simple web interface for uploading documents and
  asking questions without the command line (see bonus section below,
  if included in this submission).
- **Conversation memory:** support multi-turn follow-up questions that
  reference earlier answers in the same session.

---

## 10. Deployment Notes

This is primarily a local CLI tool. If a Streamlit bonus UI is included
in this repository, it can be deployed for free via
[Streamlit Community Cloud](https://streamlit.io/cloud) by connecting
your GitHub repo and adding `GEMINI_API_KEY` as a secret in the app
settings (never commit your real key to the repository).

---

## 11. Project Structure

```
document-qa-bot/
├── .env                  # Your real API key (never committed — gitignored)
├── .env.example          # Template showing the required variable
├── .gitignore
├── README.md              # This file
├── requirements.txt
├── data/                  # Source documents (PDF, DOCX, TXT)
├── db/                    # Persistent ChromaDB vector store (auto-generated)
└── src/
    ├── __init__.py
    ├── config.py          # Centralized constants
    ├── ingest.py          # Extraction, chunking, embedding, persistence
    ├── query.py           # Retrieval, grounded prompting, generation
    └── main.py            # Interactive CLI
```
