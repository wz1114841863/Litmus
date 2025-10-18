import sqlite3
import fitz
import re
import config
from pathlib import Path

DB_PATH = config.DB_PATH
PDF_DIR = config.PDF_DIR


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
                file_path TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    print("Database setup complete.")


def extract_info_from_pdf(pdf_path):
    """Extracts title, authors, abstract, conference, year from a PDF file."""
    try:
        doc = fitz.open(pdf_path)

        metadata = doc.metadata
        title = metadata.get("title", "N/A")
        authors = metadata.get("author", "N/A")

        if not title or title == "N/A":
            title = pdf_path.stem.replace("_", " ").replace("-", " ")

        abstract = ""
        found_abstract = False
        for page_num in range(min(2, doc.page_count)):
            page_text = doc[page_num].get_text("text").lower()
            match = re.search(
                r"abstract\s*\n([\s\S]+?)(\n\s*(1|i)\.?\s*introduction)?$",
                page_text,
                re.IGNORECASE | re.MULTILINE,
            )
            if match:
                abstract = match.group(1).strip().replace("\n", " ")
                found_abstract = True
                break

        if not found_abstract:
            abstract = "N/A"

        try:
            conference, year_str = pdf_path.parent.name.split("_")
            year = int(year_str)
        except ValueError:
            conference, year = "Unknown", 0

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "conference": conference,
            "year": year,
            "file_path": str(pdf_path.resolve()),
        }
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def insert_paper_to_db(paper_info):
    """Inserts the extracted paper information into the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM papers WHERE file_path = ?", (paper_info["file_path"],)
        )
        if cursor.fetchone() is not None:
            print(f"Paper {paper_info['file_path']} already exists in the database.")
            return False
        cursor.execute(
            """
                INSERT INTO papers (title, authors, abstract, conference, year, file_path)
                VALUES (:title, :authors, :abstract, :conference, :year, :file_path)
            """,
            paper_info,
        )
        conn.commit()
        return True


def main():
    """Main function to process all PDF files in the data directory."""
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
    for pdf_file in pdf_files:
        processed_count += 1
        print(f"Processing ({processed_count}/{len(pdf_files)}): {pdf_file}")
        paper_info = extract_info_from_pdf(pdf_file)
        if paper_info:
            was_inserted = insert_paper_to_db(paper_info)
            if was_inserted:
                new_papers += 1
                print(f"Inserted new paper: {paper_info['title']}")
            else:
                print(f"Paper already exists: {paper_info['title']}")

    print(f"\n-------Finished Data Ingestion-------")
    print(f"Total PDF files processed: {len(pdf_files)}")
    print(f"New papers added to database: {new_papers}")


if __name__ == "__main__":
    main()
