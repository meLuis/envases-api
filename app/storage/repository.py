from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.config import STORAGE_DIR


def new_dataset_id() -> str:
    return uuid4().hex[:12]


def dataset_dir(dataset_id: str) -> Path:
    return STORAGE_DIR / dataset_id


def ensure_dataset_dir(dataset_id: str) -> Path:
    path = dataset_dir(dataset_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_dataset_dir(dataset_id: str) -> Path:
    path = dataset_dir(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_id}")
    return path


def write_csv(dataset_id: str, name: str, frame: pd.DataFrame) -> str:
    path = ensure_dataset_dir(dataset_id) / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return name


def read_csv(dataset_id: str, name: str) -> pd.DataFrame:
    return pd.read_csv(require_dataset_dir(dataset_id) / name, encoding="utf-8-sig", dtype=str)


def write_json(dataset_id: str, name: str, payload: dict) -> str:
    path = ensure_dataset_dir(dataset_id) / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return name


def read_json(dataset_id: str, name: str) -> dict:
    path = require_dataset_dir(dataset_id) / name
    return json.loads(path.read_text(encoding="utf-8"))


def list_all_datasets() -> list[dict]:
    if not STORAGE_DIR.exists():
        return []
    return [
        {"dataset_id": d.name, "size_bytes": sum(f.stat().st_size for f in d.rglob("*") if f.is_file())}
        for d in sorted(STORAGE_DIR.iterdir())
        if d.is_dir()
    ]


def delete_dataset(dataset_id: str) -> None:
    path = dataset_dir(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_id}")
    shutil.rmtree(path)


def list_artifact_files(dataset_id: str) -> list[Path]:
    path = require_dataset_dir(dataset_id)
    return sorted(item for item in path.iterdir() if item.is_file())

