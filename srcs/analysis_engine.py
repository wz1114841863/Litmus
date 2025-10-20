import argparse
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


def add_column_if_not_exists(cursor, table, col_def):
    """Adds a column to a table if it does not already exist."""
    col_name = col_def.split()[0]
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [info[1] for info in cursor.fetchall()]
    if col_name not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")


def update_database_schema():
    """Adds new columns to the papers table for storing AI-generated data."""
    print("Updating database schema...")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        try:
            add_column_if_not_exists(cursor, "papers", "generated_summary TEXT")
            add_column_if_not_exists(cursor, "papers", "keywords TEXT")
            # Add a flag to track whether a paper has been analyzed
            add_column_if_not_exists(cursor, "papers", "is_analyzed INTEGER DEFAULT 0")
        except sqlite3.OperationalError as e:
            # This error occurs if the columns already exist, which is fine.
            if "duplicate column name" in str(e):
                print("  - Columns already exist, skipping.")
            else:
                raise e
        conn.commit()


def get_unprocessed_papers(re_analyze_all: bool = False):
    """Retrieves papers that have not yet been analyzed by the AI model."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if re_analyze_all:
            cursor.execute("SELECT * FROM papers")
        else:
            cursor.execute("SELECT * FROM papers WHERE is_analyzed = 0")
        return cursor.fetchall()


def generate_summary_and_keywords(abstract):
    """Uses the AI model to generate a summary and keywords for the given abstract."""
    prompt = f"""
        You are an expert AI research assistant. Your task is to analyze the following academic text (abstract + introduction).
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


def generate_conceptual_keywords_api(text):
    """Generates conceptual keywords using the AI model."""
    # [FIXED]Prompt now asks for a JSON object containing the list.
    prompt = f"""
        You are an expert AI research assistant. Your task is to extract 5-7 conceptual keywords or phrases that categorize the following academic text. Examples include "Image Segmentation", "Contrastive Learning", "Transformer Architecture", etc.
        Provide your output as a valid JSON object with a single key "keywords", which contains a list of strings.

        Text:"{text}"

        JSON Output:
    """
    if not USE_API_FOR_LLM or not API_KEY:
        raise ValueError("API key is not configured or API usage is disabled.")

    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that strictly outputs JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        reply_content = response.choices[0].message.content
        # [FIXED]Parse the object and then extract the list.
        result = json.loads(reply_content)
        return result["keywords"]
    except Exception as e:
        print(f"Error during OpenAI API call for keywords: {e}")
        return []


def generate_summary_api(text):
    """Generates a concise summary from the given text using the AI model."""
    prompt = f"""
    Based *only* on the provided academic text, generate a concise, one-sentence summary of the paper's core contribution.
    Provide your output as a valid JSON object with a single key "summary".

    Text: "{text}"

    JSON Output:
    """
    if not USE_API_FOR_LLM or not API_KEY:
        raise ValueError("OpenAI API key is not configured or API usage is disabled.")
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        reply_content = response.choices[0].message.content.strip()
        result = json.loads(reply_content)
        return result["summary"]
    except Exception as e:
        print(f"Error during OpenAI API call for summary: {e}")
        return "N/A"


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
    parser = argparse.ArgumentParser(description="Litmus AI Analysis Engine")
    parser.add_argument(
        "--re-analyze",
        action="store_true",  # 当出现 --re-analyze 时,该参数值为 True
        help="Force re-analysis of all papers in the database, ignoring the 'is_analyzed' flag.",
    )
    args = parser.parse_args()

    print("Litmus: Starting AI Analysis Engine Started...")
    update_database_schema()

    print("\n Loading models")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name="papers")
    print("Models and vector database loaded.")

    papers_to_process = get_unprocessed_papers(re_analyze_all=args.re_analyze)
    if not papers_to_process:
        print("No unprocessed papers found. All papers are up-to-date.")
        print("--- Analysis Complete ---")
        return

    for i, paper in enumerate(papers_to_process):
        paper_id = paper["id"]
        content = paper["abstract"]
        author_keywords_str = (
            paper["author_keywords"] if "author_keywords" in paper else ""
        )
        print(f"\nProcessing Paper ID {paper_id} ({i+1}/{len(papers_to_process)})")
        # if USE_API_FOR_LLM:
        #     summary, keywords = generate_summary_and_keywords_openai(abstract)
        # else:
        #     summary, keywords = generate_summary_and_keywords(abstract)

        if USE_API_FOR_LLM:
            summary = generate_summary_api(content)
            author_kws = (
                [kw.strip() for kw in author_keywords_str.split(";") if kw.strip()]
                if author_keywords_str
                else []
            )
            generative_kws = generate_conceptual_keywords_api(content)
            all_keywords = {
                "author": list(set(author_kws)),
                "generative": list(set(generative_kws)),
            }
            keywords_json = json.dumps(all_keywords, indent=2)
        else:
            print("TODO: Local LLM generation not implemented yet.")
            return

        print(f" -Generated Summary: {summary}")

        print(" -Creating vector embeddings... ")
        embeddings = embedding_model.encode(content, convert_to_tensor=False)
        collection.add(
            embeddings=[embeddings.tolist()],
            metadatas=[{"paper_id": paper_id}],
            ids=[str(paper_id)],
        )
        print(" -Vector added to ChromaDB.")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                    UPDATE papers
                    SET generated_summary = ?, keywords = ?, is_analyzed = 1
                    WHERE id = ?
                """,
                (summary, keywords_json, paper_id),
            )
            conn.commit()
        print("  - Main database updated.")

    print("\n--- Analysis Complete ---")
    print(f"Successfully processed {len(papers_to_process)} papers.")


if __name__ == "__main__":
    main()
