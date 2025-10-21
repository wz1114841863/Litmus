import argparse
import sqlite3
import json
import openai
import chromadb

import config

from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm


def add_column_if_not_exists(cursor, table, col_def):
    """Adds a column to a table if it does not already exist."""
    col_name = col_def.split()[0]
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [info[1] for info in cursor.fetchall()]
    if col_name not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")


def update_database_schema():
    """Updates the database schema to include new columns for analysis results."""
    print("Updating database schema...")
    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.cursor()

        try:
            add_column_if_not_exists(cursor, "chunks", "is_analyzed INTEGER DEFAULT 0")
            add_column_if_not_exists(cursor, "papers", "structured_summary TEXT")
            add_column_if_not_exists(cursor, "papers", "keywords TEXT")
        except sqlite3.OperationalError as e:
            pass
        conn.commit()


def get_unprocessed_chunks(re_analyze_all: bool = False):
    """Retrieves chunks that have not yet been analyzed by the AI model."""
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if re_analyze_all:
            cursor.execute(
                "SELECT c.id, c.paper_id, c.chunk_text, c.chunk_index, p.title FROM chunks c JOIN papers p ON c.paper_id = p.id"
            )
        else:
            cursor.execute(
                "SELECT c.id, c.paper_id, c.chunk_text, c.chunk_index, p.title FROM chunks c JOIN papers p ON c.paper_id = p.id WHERE c.is_analyzed = 0"
            )
        return cursor.fetchall()


def generate_conceptual_keywords_api(full_text):
    """Generates conceptual keywords using the AI model."""
    prompt_text = " ".join(full_text.split()[:4000])
    prompt = f"""
        You are an expert AI research assistant. Your task is to extract 5-7 conceptual keywords or phrases that categorize the following academic text. Examples include "Image Segmentation", "Contrastive Learning", "Transformer Architecture", etc.
        Provide your output as a valid JSON object with a single key "keywords", which contains a list of strings.

        Text:"{prompt_text}"

        JSON Output:
    """
    if not config.USE_API_FOR_LLM or not config.API_KEY:
        raise ValueError("API key is not configured or API usage is disabled.")

    client = openai.OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    try:
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
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
        result = json.loads(reply_content)
        return result["keywords"]
    except Exception as e:
        print(f"Error during OpenAI API call for keywords: {e}")
        return []


def generate_structured_summary_api(full_text):
    """Generates a concise summary from full text using the AI model."""
    # To avoid exceeding context limits, we'll use the first ~4000 words
    prompt_text = " ".join(full_text.split()[:4000])
    prompt = f"""
        You are an expert academic reviewer. Analyze the following text from a research paper.
        Your task is to extract the following key information and provide it in a valid JSON format.
        1.  **motivation**: What is the core problem or research gap the paper addresses?
        2.  **methodology**: What is the proposed solution, technique, or framework?
        3.  **key_results**: What are the main findings or contributions of the paper?

        Text: "{prompt_text}"

        JSON Output:
    """
    if not config.USE_API_FOR_LLM or not config.API_KEY:
        raise ValueError("OpenAI API key is not configured or API usage is disabled.")
    client = openai.OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    try:
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        reply_content = response.choices[0].message.content.strip()
        result = json.loads(reply_content)
        return result
    except Exception as e:
        print(f"Error during OpenAI API call for summary: {e}")
        return "N/A"


def main():
    """Main processing loop for analyzing papers"""
    parser = argparse.ArgumentParser(description="Litmus AI Analysis Engine")
    parser.add_argument(
        "--re-analyze",
        action="store_true",
        help="Force re-analysis of all papers in the database, ignoring the 'is_analyzed' flag.",
    )
    args = parser.parse_args()

    print("Litmus: Starting AI Analysis Engine Started...")
    update_database_schema()

    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name="papers")
    print("Models and vector database loaded.")

    chunks_to_process = get_unprocessed_chunks(re_analyze_all=args.re_analyze)
    if not chunks_to_process:
        print("No unprocessed chunks found. All chunks are up-to-date.")
        print("--- Analysis Complete ---")
        return

    processed_paper_ids = set()
    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.cursor()
        for chunk in tqdm(chunks_to_process, desc="Analyzing Chunks"):
            chunk_id = chunk["id"]
            paper_id = chunk["paper_id"]

            embedding = embedding_model.encode(
                chunk["chunk_text"], convert_to_tensor=False
            )

            collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[
                    {
                        "paper_id": paper_id,
                        "chunk_index": chunk["chunk_index"],
                        "paper_title": chunk["title"],
                    }
                ],
                ids=[str(chunk_id)],
            )

            cursor.execute(
                "UPDATE chunks SET is_analyzed = 1 WHERE id = ?", (chunk_id,)
            )

            processed_paper_ids.add(paper_id)
        conn.commit()
    print(f"\nSuccessfully vectorized {len(chunks_to_process)} chunks.")

    if not processed_paper_ids:
        print("No papers to summarize or extract keywords from.")
    else:
        print(
            f"\nVerifying completion status for {len(processed_paper_ids)} candidate papers..."
        )
        papers_ready_for_summary = []
        with sqlite3.connect(config.DB_PATH) as conn:
            cursor = conn.cursor()
            for paper_id in processed_paper_ids:
                # Check if all chunks for this paper are analyzed
                cursor.execute(
                    "SELECT COUNT(*) FROM chunks WHERE paper_id = ? AND is_analyzed = 0",
                    (paper_id,),
                )
                unprocessed_count = cursor.fetchone()[0]
                if unprocessed_count == 0:
                    papers_ready_for_summary.append(paper_id)

    if not papers_ready_for_summary:
        print(
            "\nNo papers have been fully processed yet. Summaries will be generated on a future run."
        )
    else:
        print(
            f"\nFound {len(papers_ready_for_summary)} papers fully processed. Generating summaries & keywords..."
        )
        with sqlite3.connect(config.DB_PATH) as conn:
            cursor = conn.cursor()
            for paper_id in tqdm(
                papers_ready_for_summary, desc="Summarizing & Tagging Papers"
            ):
                cursor.execute(
                    "SELECT full_text, title, author_keywords FROM papers WHERE id = ?",
                    (paper_id,),
                )
                result = cursor.fetchone()
                if result:
                    full_text, title, author_keywords_str = result
                    print(f"\n  - Processing paper: {title[:60]}...")

                    summary_json = generate_structured_summary_api(full_text)
                    author_kws = (
                        [
                            kw.strip()
                            for kw in author_keywords_str.split(";")
                            if kw.strip()
                        ]
                        if author_keywords_str
                        else []
                    )
                    generative_kws = generate_conceptual_keywords_api(full_text)
                    all_keywords = {
                        "author": list(set(author_kws)),
                        "generative": list(set(generative_kws)),
                    }
                    keywords_json = json.dumps(all_keywords, indent=2)

                    cursor.execute(
                        "UPDATE papers SET structured_summary = ?, keywords = ? WHERE id = ?",
                        (summary_json, keywords_json, paper_id),
                    )
            conn.commit()

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
