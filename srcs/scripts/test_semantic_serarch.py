import sqlite3
import chromadb
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config

from search_engine import semantic_search
from sentence_transformers import SentenceTransformer


def main(query="accelerator"):
    print("--- Litmus Semantic Search Diagnostic Tool ---")
    print(f"\n[CONFIG] Using Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"[CONFIG] Using ChromaDB Path: {config.CHROMA_PATH}")
    print(f"[CONFIG] Using Distance Threshold: {config.SEMANTIC_SEARCH_THRESHOLD}")
    print(f'\n[TEST] Using Query: "{query}"')

    print("\n--- Loading Models and Databases ---")
    try:
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        collection = chroma_client.get_collection(name="papers_chunks")
        print("Models and ChromaDB loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load resources: {e}")
        exit()

    print("\n--- Database Sanity Check ---")
    try:
        chunk_count = collection.count()
        print(
            f"ChromaDB collection 'paper_chunks' contains {chunk_count} chunk vectors."
        )
        if chunk_count == 0:
            print(
                "CRITICAL ERROR: No chunks found in the vector database. Please run the AI analysis script first."
            )
            exit()

        # check distance metric, ensure it's cosine
        collection_metadata = collection.metadata
        distance_metric = collection_metadata.get("hnsw:space")
        print(f"ChromaDB distance metric is set to: '{distance_metric}'.")
        if distance_metric != "cosine":
            print(
                "CRITICAL WARNING: Distance metric is NOT 'cosine'. This is likely the root cause of the problem."
            )
            print(
                "   Please run the full reset and rebuild process from the previous step."
            )
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to check database status: {e}")
        exit()

    print("\n--- Performing Raw Semantic Search ---")
    try:
        query_with_instruction = (
            f"Represent this sentence for searching relevant passages: {query}"
        )
        query_embedding = embedding_model.encode(
            query_with_instruction,
            convert_to_tensor=False,
        ).tolist()

        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["metadatas", "documents", "distances"],
        )
        # print("Raw results:", raw_results)
        print("Raw query to ChromaDB completed. Analyzing results...")

        if not raw_results["ids"][0]:
            print("CRITICAL ERROR: Raw query returned ZERO results from ChromaDB.")
        else:
            for i in range(len(raw_results["ids"][0])):
                distance = raw_results["distances"][0][i]
                document = raw_results["documents"][0][i]
                paper_id = raw_results["metadatas"][0][i]["paper_id"]

                print(f"\n  --- Raw Result #{i+1} ---")
                print(f"  Distance: {distance:.4f}")
                print(f"  Paper ID: {paper_id}")
                print(f'  Chunk Text: "...{document[:200].strip()}..."')

                # 分析距离值
                if distance < config.SEMANTIC_SEARCH_THRESHOLD:
                    print("  ANALYSIS: This result PASSES the distance threshold.")
                else:
                    print(
                        f"  ANALYSIS: This result FAILS the distance threshold (Distance > {config.SEMANTIC_SEARCH_THRESHOLD})."
                    )

    except Exception as e:
        print(f"CRITICAL ERROR: An error occurred during the raw query: {e}")

    print("\n--- Testing the Application's `semantic_search_chunks` function ---")
    try:

        app_results = semantic_search(query, top_k=5)

        print(f"Application function returned {len(app_results)} results.")

        if app_results:
            print("   --- Filtered Results ---")
            for i, chunk_info in enumerate(app_results):
                print(
                    f"   #{i+1}: Paper ID {chunk_info['paper_id']}, Distance {chunk_info['distance']:.4f}"
                )
        else:
            print("   ANALYSIS: The application logic filtered out all raw results.")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to run the application's search function: {e}")

    print("\n--- Diagnostic Complete ---")


if __name__ == "__main__":
    """
    Example usage:
        python test_semantic_search.py

    """
    test_query = "accelerator"
    main(test_query)
