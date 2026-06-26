from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


def read_table(file: BinaryIO | str | Path, filename: str | None = None) -> pd.DataFrame:
    name = (filename or str(getattr(file, "name", file))).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, sheet_name=0, dtype=str)
    for sep in (",", ";", "\t", "|"):
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            frame = pd.read_csv(file, sep=sep, encoding="utf-8-sig", dtype=str)
            if len(frame.columns) > 1:
                return frame
        except Exception:
            continue
    if hasattr(file, "seek"):
        file.seek(0)
    return pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig", dtype=str)

