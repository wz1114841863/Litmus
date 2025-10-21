import argparse
import chromadb
import sqlite3
import fitz  # PyMuPDF
import re
import shutil
import os

import config

from pathlib import Path
from tqdm import tqdm


def setup_database():
    """Creates the papers and chunks table in the database."""
    print("Setting up database...")
    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                authors TEXT,
                conference TEXT,
                year INTEGER,
                file_path TEXT NOT NULL UNIQUE,
                author_keywords TEXT,
                full_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()
    print("Database setup complete.")


def reset_database():
    """Deletes the existing database file to reset the database."""
    if os.path.exists(config.DB_PATH):
        os.remove(config.DB_PATH)
        print("Existing database deleted.")

    if os.path.exists(config.CHROMA_PATH):
        shutil.rmtree(config.CHROMA_PATH)
        print("Existing ChromaDB vector store deleted.")

    print("Database has been reset.")


def chunk_text(text, chunk_size=512, overlap=50):
    """Splits text into chunks of specified size with overlap."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_chunk_words = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(current_chunk_words) + len(words) <= chunk_size:
            current_chunk_words.extend(words)
        else:
            chunks.append(" ".join(current_chunk_words))
            overlap_words = current_chunk_words[-overlap:] if overlap > 0 else []
            current_chunk_words = overlap_words + words

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


def extract_info_from_pdf(pdf_path):
    """Extracts title, authors, abstract, conference, year from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        title = metadata.get("title", pdf_path.stem.replace("_", " "))
        authors = metadata.get("author", "N/A")  # TODO: can be improved with regex
        author_keywords = metadata.get("keywords", "")

        full_text = "".join(page.get_text("text") for page in doc)

        try:
            conference, year_str = pdf_path.parent.name.split("_")
            year = int(year_str)
        except ValueError:
            print(
                f"\n - Warning: Could not parse conference and year from folder name '{pdf_path.parent.name}'. Setting as 'Unknown' and 0."
            )
            conference, year = "Unknown", 0

        return {
            "title": title,
            "authors": authors,
            "conference": conference,
            "year": year,
            "file_path": str(pdf_path.resolve()),
            "author_keywords": author_keywords,
            "full_text": full_text,
        }

    except Exception as e:
        error_message = f"Failed to parse {pdf_path.name}: {e}"
        print(f"\n  - Error: {error_message}")
        with open(config.LOG_FILE_PATH, "a") as f:
            f.write(f"{pdf_path.resolve()} - {error_message}\n")
        return None


def insert_paper_to_db(paper_info, cursor):
    """Inserts the extracted paper information into the database."""

    cursor.execute(
        "SELECT id FROM papers WHERE file_path = ?", (paper_info["file_path"],)
    )

    if cursor.fetchone() is not None:
        print(f"Paper {paper_info['file_path']} already exists in the database.")
        return False

    cursor.execute(
        """
            INSERT INTO papers (title, authors, conference, year, file_path, author_keywords, full_text)
            VALUES (:title, :authors, :conference, :year, :file_path, :author_keywords, :full_text)
        """,
        paper_info,
    )
    return True


def sync_database_with_filesystem():
    """Removes database entries for papers whose PDF files no longer exist."""
    print("Syncing database with filesystem...")
    filesystem_paths = {str(pdf.resolve()) for pdf in config.PDF_DIR.rglob("**/*.pdf")}
    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path FROM papers")
        db_papers = {row[1]: row[0] for row in cursor.fetchall()}
        db_paths = set(db_papers.keys())

        paths_to_delete = db_paths - filesystem_paths

        if not paths_to_delete:
            print("No database entries to remove. Database is in sync.")
            return

        print(f"Removing {len(paths_to_delete)} entries from database...")
        ids_to_delete = [db_papers[path] for path in paths_to_delete]

        placeholders = ", ".join("?" for _ in ids_to_delete)
        cursor.execute(
            f"DELETE FROM papers WHERE id IN ({placeholders})", ids_to_delete
        )
        conn.commit()

        # Remove from ChromaDB as well
        try:
            chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
            collection = chroma_client.get_or_create_collection(name="papers")
            collection.delete(ids=[str(paper_id) for paper_id in ids_to_delete])
            print("Removed corresponding entries from ChromaDB.")
        except Exception as e:
            print(f"Error removing entries from ChromaDB: {e}")
    print("Database sync complete.")


def main():
    """Main function to process all PDF files in the data directory."""
    parser = argparse.ArgumentParser(description="Litmus Data Ingestion Script")
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    args = parser.parse_args()
    if args.reset:
        reset_database()

    print("Litmus: Starting data ingestion...")
    setup_database()

    print(f"Processing PDF folders {config.PDF_DIR}...")
    pdf_files = list(config.PDF_DIR.rglob("**/*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        print("Directory Structure: data/pdfs/<Conference>_<Year>/<paper_files>.pdf")
        print("Example: data/pdfs/ICML_2023/sample_paper.pdf")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting extraction...")

    new_papers = 0
    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.cursor()
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_file = Path(pdf_path)
            paper_info = extract_info_from_pdf(pdf_file)
            if paper_info:
                was_inserted = insert_paper_to_db(paper_info, cursor)
                paper_id = cursor.lastrowid if was_inserted else None
                chunks = chunk_text(paper_info["full_text"])
                chunk_sql = "INSERT INTO chunks (paper_id, chunk_text, chunk_index) VALUES (?, ?, ?)"
                chunk_data = [(paper_id, text, i) for i, text in enumerate(chunks)]
                cursor.executemany(chunk_sql, chunk_data)

                new_papers += 1
        conn.commit()

    print(f"\n-------Finished Data Ingestion-------")
    print(f"New papers added to database: {new_papers}")

    # Sync database with filesystem
    sync_database_with_filesystem()


if __name__ == "__main__":
    main()
