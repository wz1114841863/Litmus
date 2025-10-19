import sqlite3
import json
import openai
import config
import chromadb

from sentence_transformers import SentenceTransformer
from pathlib import Path

DB_PATH = config.DB_PATH
CHROMA_PATH = config.CHROMA_PATH
EMBEDDING_MODEL = config.EMBEDDING_MODEL
LLM_MODEL = config.LLM_MODEL
USE_API_FOR_LLM = config.USE_API_FOR_LLM
API_KEY = config.OPENAI_API_KEY
BASE_URL = config.BASE_URL
MODEL_NAME = config.MODEL_NAME


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


def generate_summary_and_keywords(abstract):
    """Uses the AI model to generate a summary and keywords for the given abstract."""
    prompt = f"""
        You are an expert AI research assistant. Your task is to analyze the following academic paper abstract.
        Based *only* on the text provided, generate:
        1. A concise, one-sentence summary of the paper's core contribution.
        2. A list of 5-7 conceptual keywords or phrases that categorize this paper. Examples include "Image Segmentation", "Contrastive Learning", "Transformer Architecture", etc.

        Do not use any information outside of the provided abstract.
        Provide your output in a valid JSON format with two keys: "summary" and "keywords".

        Abstract:
        "{abstract}"

        JSON Output:
    """
    """Requires local deployment of the corresponding model implementation."""
    pass
    return "N/A", []


def generate_summary_and_keywords_openai(abstract):
    """Uses OpenAI API to generate a summary and keywords for the given abstract."""
    if not USE_API_FOR_LLM or not API_KEY:
        raise ValueError("OpenAI API key is not configured or API usage is disabled.")

    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = f"""
        You are an expert AI research assistant. Your task is to analyze the following academic paper abstract.
        Based *only* on the text provided, generate:
        1. A concise, one-sentence summary of the paper's core contribution.
        2. A list of 5-7 conceptual keywords or phrases that categorize this paper. Examples include "Image Segmentation", "Contrastive Learning", "Transformer Architecture", etc.

        Do not use any information outside of the provided abstract.
        Provide your output in a valid JSON format with two keys: "summary" and "keywords".

        Abstract:
        "{abstract}"

        JSON Output:
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # maybe need a Config option
        )
        reply_content = response.choices[0].message.content.strip()
        result = json.loads(reply_content)
        return result["summary"], result["keywords"]
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return "N/A", []


def main():
    """Main processing loop for analyzing papers"""
    print("Litmus: Starting AI Analysis Engine Started...")
    update_database_schema()

    print("\n Loading models")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name="papers")
    print("Models and vector database loaded.")

    papers_to_process = get_unprocessed_papers()
    if not papers_to_process:
        print("No unprocessed papers found. All papers are up-to-date.")
        print("--- Analysis Complete ---")
        return

    for i, paper in enumerate(papers_to_process):
        paper_id = paper["id"]
        abstract = paper["abstract"]
        print(f"\nProcessing Paper ID {paper_id} ({i+1}/{len(papers_to_process)})")
        if USE_API_FOR_LLM:
            summary, keywords = generate_summary_and_keywords_openai(abstract)
        else:
            summary, keywords = generate_summary_and_keywords(abstract)
        keywords_str = ", ".join(keywords)
        print(f"  - Summary: {summary}")
        print(f"  - Keywords: {keywords_str}")

        print(" -Creating vector embeddings... ")
        embeddings = embedding_model.encode(abstract, convert_to_tensor=False)

        collection.add(
            embeddings=[embeddings.tolist()],
            metadatas=[{"paper_id": paper_id}],  # 原来是字符串,现在改成 dict
            ids=[str(paper_id)],
        )
        print(" -Paper processed and added to database.")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                    UPDATE papers
                    SET generated_summary = ?, keywords = ?, is_analyzed = 1
                    WHERE id = ?
                """,
                (summary, keywords_str, paper_id),
            )
            conn.commit()
        print("  - Main database updated.")

    print("\n--- Analysis Complete ---")
    print(f"Successfully processed {len(papers_to_process)} papers.")


if __name__ == "__main__":
    main()
