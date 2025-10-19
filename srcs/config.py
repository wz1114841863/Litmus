from pathlib import Path
from api_keys import API_KEY

# Project root directory (Litmus/)
BASE_DIR = Path(__file__).parent.parent

# Data directory (Litmus/data/)
DATA_DIR = BASE_DIR / "data"

# PDF storage directory (Litmus/data/pdfs/)
PDF_DIR = DATA_DIR / "pdfs"

# SQLite database file path
DB_PATH = DATA_DIR / "metadata.sqlite"

# ChromaDB vector database storage path
CHROMA_PATH = DATA_DIR / "vector_store"


# --- AI Model Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3:8b"

# --- API Configuration ---
USE_API_FOR_LLM = True  # Set to True to switch to API mode
OPENAI_API_KEY = API_KEY
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"  # OpenAI model name
