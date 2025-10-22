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
        # print(f"  - Keywords response: {reply_content}")
        result = json.loads(reply_content)
        return result["keywords"]
    except Exception as e:
        print(f"Error during OpenAI API call for keywords: {e}")
        return []


def generate_structured_summary_api(full_text):
    """Generates a concise summary from full text using the AI model."""
    # To avoid exceeding context limits, we'll use the first ~4000 words
    prompt_text = " ".join(full_text.split()[:4000])
    prompt = f"""You are an expert academic reviewer.
        Extract three items from the paper excerpt and return **only** valid JSON (no markdown fences, no extra text).

        Required JSON structure:
        {{
            "motivation":   "core problem or research gap the paper addresses",
            "methodology":  "proposed solution, technique, or framework",
            "key_results":  "main findings, performance gains, or contributions"
        }}

        Paper excerpt:
        {prompt_text}

        JSON:
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
        # print(f"[SUMMARY_RAW] len={len(reply_content)}  repr={reply_content!r}")
        cleaned = reply_content.strip()
        for fence in ("```json", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence) :]
            if cleaned.endswith(fence):
                cleaned = cleaned[: -len(fence)]
        cleaned = cleaned.strip()
        result = json.loads(cleaned)  # ensure valid JSON
        return json.dumps(result, ensure_ascii=False)
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
    parser.add_argument(
        "--re-summary",
        action="store_true",
        help="Skip embedding; re-generate summaries & keywords for fully-vectorized papers.",
    )
    args = parser.parse_args()

    print("Litmus: Starting AI Analysis Engine Started...")
    update_database_schema()

    if args.re_summary:
        embedding_model = None
        collection = None
        chroma_client = None
    else:
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        collection = chroma_client.get_or_create_collection(
            name="papers_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        print("Models and vector database loaded.")

    if not args.re_summary:
        chunks_to_process = get_unprocessed_chunks(re_analyze_all=args.re_analyze)
        if not chunks_to_process:
            print("No unprocessed chunks found. All chunks are up-to-date.")
            print("--- Analysis Complete ---")
            return

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
                    documents=[chunk["chunk_text"]],
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
            conn.commit()
        print(f"\nSuccessfully vectorized {len(chunks_to_process)} chunks.")

    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT paper_id "
            "FROM chunks "
            "GROUP BY paper_id "
            "HAVING SUM(CASE WHEN is_analyzed = 0 THEN 1 ELSE 0 END) = 0"
        )
        papers_ready_for_summary = [row[0] for row in cursor.fetchall()]

    if not papers_ready_for_summary:
        print("\nNo papers are ready for summarization.")
        return

    print(f"\n{len(papers_ready_for_summary)} papers ready for summarization.")

    for paper_id in tqdm(papers_ready_for_summary, desc="Summarizing & Tagging Papers"):
        with sqlite3.connect(config.DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT full_text, title, author_keywords FROM papers WHERE id = ?",
                (paper_id,),
            )
            full_text, title, author_keywords_str = cursor.fetchone() or (
                None,
                None,
                None,
            )
            if not full_text:
                continue

            # print(f"\n  - Processing paper: {title[:60]}...")

            try:
                summary_json = generate_structured_summary_api(full_text)
                author_kws = [
                    kw.strip()
                    for kw in (author_keywords_str or "").split(";")
                    if kw.strip()
                ]
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
            except Exception as e:
                print(f"    [!] Failed on paper {paper_id}: {e}")
                continue

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
