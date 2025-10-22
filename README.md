# Litmus 🧪

*Your Personal AI Research Assistant for Novelty Checking*

## Core Idea

In academic research, one of the most critical tasks is validating the novelty of an idea. Traditional search engines are often too broad, and large language models can hallucinate or provide incomplete results.

**Litmus** is designed to solve this problem. It is a local-first, AI-powered tool that allows you to build a curated, reliable, and searchable knowledge base from your own collection of academic papers (e.g., proceedings from top-tier conferences like CVPR, NeurIPS, HPCA).

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

## Env Setup
1. python Env Setup
```
# uv is recommended. You can also use conda.
uv venv --python 3.12
source ./.venv/bin/activate
uv pip install -r requirements.txt
```

2. Prepare Conference Papers
Locate the `data/pdfs/` directory within your project.
Within this directory, create subdirectories using the format
`conference_name_year`. For example: `CVPR_2024`.
Place the PDF papers you have collected into the corresponding directories.

3. Create srcs/api_keys.py, and write
```API_KEY = "your-open_api-key"```

4. create a new sql database, use --reset means delete old database
    ```python srcs/data_ingestion.py --reset```
or
    ```python srcs/data_ingestion.py```

5. generate vector and save in to chromadb, it will take a long time. But it executes only once.
    ```python srcs/analysis_engine --re-analyze``` or ```python srcs/analysis_engine```

6. Start Web service
    ```streamlit run srcs/0_🔎_Search.py```
---
*This project is currently under active development.*
