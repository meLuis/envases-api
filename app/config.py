from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage" / "datasets")))

MIN_PRODUCT_ROWS = 10
MIN_TRANSACTION_ROWS = 10
MIN_ATTRIBUTE_COVERAGE = 0.35
MIN_GRAPH_EDGES = 5
MIN_UFDS_SIMILARITY = 0.75

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://cheerful-beignet-067d26.netlify.app",
    ).split(",")
    if origin.strip()
]
