"""
main.py

Interactive command-line interface for the Document Q&A RAG bot.

This is the "front door" of the application -- the script an end user
(or a grader) actually runs. It ties together ingest.py (Phase 2) and
query.py (Phase 3) behind a simple text menu:

    1. Ingest documents (build/refresh the vector database)
    2. Ask a question (repeatable -- loops until the user exits)
    3. Exit

Run from the project root:
    python -m src.main
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from google import genai

from src import config
from src.ingest import run_ingestion
from src.query import query_rag_pipeline, load_collection

load_dotenv()


# ==========================================================================
# Display helpers
# ==========================================================================

def print_banner() -> None:
    """Prints the welcome banner shown once at startup."""
    print("=" * 60)
    print("  Document Q&A Bot — RAG-powered, grounded in your files")
    print("=" * 60)


def print_menu() -> None:
    """Prints the main menu options."""
    print("\nWhat would you like to do?")
    print("  [1] Ingest documents (scan data/ and (re)build the database)")
    print("  [2] Ask a question")
    print("  [3] Exit")


def print_answer(answer: str, citations: list[str], show_sources: bool) -> None:
    """
    Prints a query result in a clean, readable format.

    Args:
        answer: The generated answer text.
        citations: List of human-readable citation strings.
        show_sources: Whether to print the citations section. Lets the
            CLI offer "show retrieved citations separately" as an
            optional toggle, per the assignment's CLI requirements.
    """
    print("\n" + "-" * 60)
    print("ANSWER:")
    print(answer)

    if show_sources:
        print("\nSOURCES:")
        if citations:
            for c in citations:
                print(f"  - {c}")
        else:
            print("  (no sources retrieved)")
    print("-" * 60)


# ==========================================================================
# Menu actions
# ==========================================================================

def handle_ingest() -> None:
    """
    Runs the ingestion pipeline (Phase 2) from within the CLI.

    Wrapped in a try/except so a failure (bad API key, no files in
    data/, network issue) shows a clean message and returns to the
    menu instead of crashing the whole CLI session.
    """
    print("\nStarting ingestion. This scans data/, extracts text, "
          "chunks it, and embeds it via the Gemini API -- this may "
          "take a little while and will call the API for every chunk.\n")
    try:
        run_ingestion()
    except SystemExit:
        # run_ingestion() calls sys.exit() on fatal errors (e.g. no files
        # found, missing API key). Catch that here so the CLI keeps running
        # instead of the whole process exiting.
        print("\nIngestion did not complete. See the messages above for why.")
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed unexpectedly: {e}")


def handle_ask(client: genai.Client) -> None:
    """
    Runs the "ask questions" loop: repeatedly prompts for a question
    and prints grounded answers until the user types a way to stop.

    Args:
        client: A shared, already-authenticated google-genai Client,
            reused across every question in this session so we don't
            re-create a new client (and re-validate the API key) on
            every single question.
    """
    # Check the database exists/has content before entering the loop,
    # so we give one clear message instead of repeating it per question.
    try:
        collection = load_collection(client)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        return

    if collection.count() == 0:
        print("\nThe database is empty. Run option [1] to ingest documents first.")
        return

    print("\nAsk away! Type 'back' or 'menu' to return to the main menu.\n")

    while True:
        question = input("Your question: ").strip()

        if not question:
            print("Please type a question (or 'back' to return to the menu).")
            continue

        if question.lower() in {"back", "menu", "exit", "quit"}:
            break

        try:
            result = query_rag_pipeline(question, client=client)
        except RuntimeError as e:
            print(f"\n[ERROR] {e}")
            continue

        show_sources_input = input(
            "Show sources for this answer? [Y/n]: "
        ).strip().lower()
        show_sources = show_sources_input != "n"

        print_answer(result.answer, result.citations, show_sources)
        print()


# ==========================================================================
# Main loop
# ==========================================================================

def main() -> None:
    """
    Entry point: prints the banner, then loops the main menu until
    the user chooses to exit.
    """
    print_banner()

    # Validate the API key once, up front, with a clear error if missing --
    # better than discovering it's missing halfway through a menu action.
    try:
        api_key = config.get_api_key()
    except EnvironmentError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    while True:
        print_menu()
        choice = input("\nEnter your choice [1-3]: ").strip()

        if choice == "1":
            handle_ingest()
        elif choice == "2":
            handle_ask(client)
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            print(f"\n'{choice}' isn't a valid option. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
