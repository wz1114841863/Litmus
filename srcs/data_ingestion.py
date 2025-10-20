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


DB_PATH = config.DB_PATH
PDF_DIR = config.PDF_DIR
CHROMA_PATH = config.CHROMA_PATH
LOG_FILE_PATH = config.LOG_FILE_PATH


def setup_database():
    """Creates the papers table in the database if it doesn't already exist."""
    print("Setting up database...")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                authors TEXT,
                abstract TEXT,
                conference TEXT,
                year INTEGER,
                pdf_path TEXT,
                author_keywords TEXT,
                file_path TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    print("Database setup complete.")


def reset_database():
    """Deletes the existing database file to reset the database."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Existing database deleted.")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Existing ChromaDB vector store deleted.")

    print("Database has been reset.")


def extract_info_from_pdf(pdf_path):
    """Extracts title, authors, abstract, conference, year from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        title = metadata.get("title", "N/A")
        authors = metadata.get("author", "N/A")

        if not title or title == "N/A":
            title = pdf_path.stem.replace("_", " ").replace("-", " ")

        full_text = ""
        for page_num in range(min(3, doc.page_count)):
            full_text += doc[page_num].get_text("text") + "\n"

        # Extract abstract using regex
        abstract_match = re.search(
            r"Abstract([\s\S]*?)(?:1\.? Introduction|I\. INTRODUCTION)",
            full_text,
            re.DOTALL | re.IGNORECASE,
        )
        abstract = (
            abstract_match.group(1).strip().replace("\n", " ")
            if abstract_match
            else "Could not automatically extract abstract."
        )

        # Extract introduction section
        intro_match = re.search(
            r"(?:1\.?\s*\n*Introduction|I\.\s*\n*INTRODUCTION)([\s\S]*?)(?:2\.?\s*\n*|II\.\s*\n*)",
            full_text,
            re.DOTALL,
        )
        introduction = (
            intro_match.group(1).strip().replace("\n", " ")
            if intro_match
            else "Could not automatically extract introduction."
        )

        # Extract keywords if available
        author_keywords = metadata.get("keywords", "")

        # Concat abstract and introduction for better context
        combined_content = f"Abstract: {abstract}\n\nIntroduction: {introduction}"

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
            "abstract": combined_content,
            "conference": conference,
            "year": year,
            "file_path": str(pdf_path.resolve()),
            "author_keywords": author_keywords,
        }

    except Exception as e:
        error_message = f"Failed to parse {pdf_path.name}: {e}"
        print(f"\n  - Error: {error_message}")
        with open(LOG_FILE_PATH, "a") as f:
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
            INSERT INTO papers (title, authors, abstract, conference, year, file_path, author_keywords)
            VALUES (:title, :authors, :abstract, :conference, :year, :file_path, :author_keywords)
        """,
        paper_info,
    )
    return True


def sync_database_with_filesystem():
    """Removes database entries for papers whose PDF files no longer exist."""
    print("Syncing database with filesystem...")
    filesystem_paths = {str(pdf.resolve()) for pdf in PDF_DIR.rglob("**/*.pdf")}
    with sqlite3.connect(DB_PATH) as conn:
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
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            collection = chroma_client.get_or_create_collection(name="papers")
            collection.delete(ids=[str(paper_id) for paper_id in ids_to_delete])
            print("Removed corresponding entries from ChromaDB.")
        except Exception as e:
            print(f"Error removing entries from ChromaDB: {e}")
    print("Database sync complete.")


def main():
    """Main function to process all PDF files in the data directory."""
    parser = argparse.ArgumentParser(description="Litmus 数据采集与管理脚本")
    parser.add_argument("--reset", action="store_true", help="清除并重建所有数据库.")
    args = parser.parse_args()
    if args.reset:
        reset_database()

    print("Litmus: Starting data ingestion...")
    setup_database()

    print(f"Processing PDF folders {PDF_DIR}...")
    pdf_files = list(PDF_DIR.rglob("**/*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        print("Directory Structure: data/pdfs/<Conference>_<Year>/<paper_files>.pdf")
        print("Example: data/pdfs/ICML_2023/sample_paper.pdf")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting extraction...")

    new_papers = 0
    processed_count = 0

    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.cursor()
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_file = Path(pdf_path)
            processed_count += 1
            # print(f"Processing ({processed_count}/{len(pdf_files)}): {pdf_file}")
            paper_info = extract_info_from_pdf(pdf_file)
            if paper_info:
                was_inserted = insert_paper_to_db(paper_info, cursor)
                if was_inserted:
                    new_papers += 1
                #     print(f"Inserted new paper: {paper_info['title']}")
                # else:
                #     print(f"Paper already exists: {paper_info['title']}")

    print(f"\n-------Finished Data Ingestion-------")
    print(f"Total PDF files processed: {len(pdf_files)}")
    print(f"New papers added to database: {new_papers}")

    # Sync database with filesystem
    sync_database_with_filesystem()


if __name__ == "__main__":
    main()
