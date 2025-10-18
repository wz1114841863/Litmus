import sqlite3
import json
import config
from pathlib import Path

DB_PATH = config.DB_PATH
CHROMA_PATH = config.CHROMA_PATH
EMBEDDING_MODEL = config.EMBEDDING_MODEL
LLM_MODEL = config.LLM_MODEL


def update_database_schema():
    """Adds new columns to the papers table for storing AI-generated data."""
    print("Updating database schema...")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        try:
            cursor.execute("ALTER TABLE papers ADD COLUMN generated_summary TEXT")
            cursor.execute("ALTER TABLE papers ADD COLUMN keywords TEXT")
            # Add a flag to track whether a paper has been analyzed
            cursor.execute(
                "ALTER TABLE papers ADD COLUMN is_analyzed INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError as e:
            # This error occurs if the columns already exist, which is fine.
            if "duplicate column name" in str(e):
                print("  - Columns already exist, skipping.")
            else:
                raise e
        conn.commit()


def get_unprocessed_papers():
    """Retrieves papers that have not yet been analyzed by the AI model."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE is_analyzed = 0")
        return cursor.fetchall()


