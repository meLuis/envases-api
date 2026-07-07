from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
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
    datasets: list[dict] = []
    for d in STORAGE_DIR.iterdir():
        if not d.is_dir():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        summary = d / "dataset_summary.json"
        loaded_ts = summary.stat().st_mtime if summary.exists() else d.stat().st_ctime
        updated_ts = max((f.stat().st_mtime for f in files), default=loaded_ts)
        datasets.append(
            {
                "dataset_id": d.name,
                "size_bytes": sum(f.stat().st_size for f in files),
                "loaded_at": datetime.fromtimestamp(loaded_ts, tz=timezone.utc).isoformat(),
                "updated_at": datetime.fromtimestamp(updated_ts, tz=timezone.utc).isoformat(),
            }
        )
    return sorted(datasets, key=lambda item: item["loaded_at"], reverse=True)


def delete_dataset(dataset_id: str) -> None:
    path = dataset_dir(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_id}")
    shutil.rmtree(path)


def list_artifact_files(dataset_id: str) -> list[Path]:
    path = require_dataset_dir(dataset_id)
    return sorted(item for item in path.iterdir() if item.is_file())

