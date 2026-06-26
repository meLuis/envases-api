from __future__ import annotations

from pathlib import Path

from app.services.pipeline import build_dataset_from_paths


BASE = Path(__file__).resolve().parent / "data" / "base"


if __name__ == "__main__":
    summary = build_dataset_from_paths(
        BASE / "productos.csv",
        BASE / "ventas.csv",
        BASE / "items_compras.csv",
    )
    print(summary.model_dump_json(indent=2))

