from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage" / "datasets")))

MIN_PRODUCT_ROWS = 10
MIN_TRANSACTION_ROWS = 10
MIN_ATTRIBUTE_COVERAGE = 0.35
MIN_GRAPH_EDGES = 5
MIN_UFDS_SIMILARITY = 0.75
MIN_SUPPLIER_PROJECTION_SIMILARITY = 0.05
MIN_SUPPLIER_UFDS_SIMILARITY = 0.08
MIN_SUPPLIER_SHARED_PRODUCTS = 2

# Genera los PNG estáticos de los grafos principales al construir el dataset.
# Se puede desactivar (ENABLE_GRAPH_IMAGES=0) si el host va justo de recursos.
ENABLE_GRAPH_IMAGES = os.getenv("ENABLE_GRAPH_IMAGES", "1").strip().lower() not in {"0", "false", "no"}

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://cheerful-beignet-067d26.netlify.app",
    ).split(",")
    if origin.strip()
]
