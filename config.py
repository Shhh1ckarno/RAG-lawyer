import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BASE_URL = os.getenv("BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT") or None
JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT") or None
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "1800"))
REWRITE_REASONING_EFFORT = os.getenv("REWRITE_REASONING_EFFORT", "none")

SOURCE_PATH = BASE_DIR / "uk_rf.txt"
DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "uk_rf_parts"
REWRITTEN_PATH = BASE_DIR / "parts_rewritten.jsonl"

SPECIAL_PART_FIRST_ARTICLE = 105
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))

if not LLM_MODEL:
    raise ValueError("Укажи LLM_MODEL в файле .env")

if not EMBED_MODEL:
    raise ValueError("Укажи EMBED_MODEL в файле .env")
