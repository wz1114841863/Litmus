# Litmus 🧪

*Your Personal AI Research Assistant for Novelty Checking*

## Core Idea

In academic research, one of the most critical tasks is validating the novelty of an idea. Traditional search engines are often too broad, and large language models can hallucinate or provide incomplete results.

**Litmus** is designed to solve this problem. It is a local-first, AI-powered tool that allows you to build a curated, reliable, and searchable knowledge base from your own collection of academic papers (e.g., proceedings from top-tier conferences like CVPR, NeurIPS, ACL).

The name "Litmus" is inspired by the litmus test, which quickly determines if a substance is acidic or alkaline. Similarly, this tool helps you perform a quick "novelty test" on your research ideas against a specific, trusted body of literature.

## Key Features

* **🔎 Local & Private**: All your papers, metadata, and analysis results are stored locally on your machine. Nothing is ever sent to the cloud.
* **📚 Focused Knowledge Base**: You control the scope. Build your library from specific conferences and years that matter most to your research field.
* **🤖 AI-Powered Analysis**: For each paper, Litmus automatically:
    * Generates concise summaries.
    * Extracts comprehensive, multi-layered keywords (author-provided, extractive, and AI-generated concepts).
    * Creates semantic vector embeddings for deep content understanding.
* **💡 Hybrid Search Engine**: Go beyond simple keyword matching. Litmus allows you to:
    * **Ask questions in natural language** to find conceptually similar papers.
    * Perform robust keyword searches for specific terms.
    * Combine both search methods for the most comprehensive results.
* **🌐 Interactive UI**: A simple, clean web interface built with Streamlit allows for easy searching, filtering, and exploration of your paper collection.

## Tech Stack

* **Backend**: Python
* **Frontend/UI**: Streamlit
* **Data Storage**: SQLite (for metadata), ChromaDB (for vector storage)
* **AI Models**:
    * **Embedding**: Sentence-Transformers (using local, open-source models like `BGE-M3`).
    * **LLM Tasks**: Locally-run models via Ollama (e.g., `Llama-3-8B`, `Qwen2-7B`).
* **Core Libraries**: PyMuPDF, Pandas, scikit-learn

## Project Vision

The goal of Litmus is not to replace large-scale search engines like Google Scholar or Semantic Scholar. Instead, it aims to be a researcher's trusted personal assistant—a tool that provides deep, reliable, and lightning-fast insights within a high-signal, low-noise environment that you create and control.

---
*This project is currently under active development.*
