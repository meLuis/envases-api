from __future__ import annotations

import re

import pandas as pd

from app.core.text import normalize_text


PRODUCT_TYPES = ["FRASCO", "POTE", "TAPA", "BOLSA", "PISETA", "PROBADOR", "PASTILLERO", "ENVASE"]
MATERIALS = ["VIDRIO", "PLASTICO", "PVC", "PEAD", "PET", "PP", "CRISTAL"]
COLORS = ["AMBAR", "BLANCO", "BLANCA", "NEGRO", "NEGRA", "TRANSPARENTE", "TRASPARENTE", "NATURAL"]
FEATURES = ["GOTERO", "ATOMIZADOR", "SPRAY", "ROSCA", "DISPENSADOR", "DISPENSADORA", "CHUPON", "TAPON", "ASA"]


def extract_product_attributes(products: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for product in products.itertuples(index=False):
        name = str(product.product_name)
        norm = normalize_text(name)
        attrs = {
            "product_id": str(product.product_id),
            "product_name": name,
            "product_type": _first_match(norm, PRODUCT_TYPES),
            "material": _first_match(norm, MATERIALS),
            "color": _first_match(norm, COLORS),
            "capacity": _capacity(norm),
            "mouth": _mouth(norm),
            "features": "|".join(feature for feature in FEATURES if feature in norm),
            "category_use": _category_use(norm),
        }
        attrs["attribute_count"] = sum(1 for key, value in attrs.items() if key not in {"product_id", "product_name"} and value)
        rows.append(attrs)
    frame = pd.DataFrame(rows)
    coverage = float((frame["attribute_count"] > 0).mean()) if not frame.empty else 0.0
    report = {
        "products": int(len(frame)),
        "products_with_attributes": int((frame["attribute_count"] > 0).sum()) if not frame.empty else 0,
        "coverage": round(coverage, 4),
        "rules": {
            "product_types": PRODUCT_TYPES,
            "materials": MATERIALS,
            "colors": COLORS,
            "features": FEATURES,
        },
    }
    return frame, report


def _first_match(norm: str, values: list[str]) -> str:
    return next((value for value in values if value in norm), "")


def _capacity(norm: str) -> str:
    match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(ML|LT|L|GR|KG)\b", norm)
    return f"{match.group(1)}{match.group(2)}" if match else ""


def _mouth(norm: str) -> str:
    match = re.search(r"\b(B|N)\s?(\d{2,3})\s?MM?\b", norm)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    match = re.search(r"\b(B|N)(\d{2,3})\b", norm)
    return f"{match.group(1)}{match.group(2)}" if match else ""


def _category_use(norm: str) -> str:
    if any(word in norm for word in ["ODONTO", "GOTERO", "PROBADOR"]):
        return "salud_cosmetica"
    if any(word in norm for word in ["POTE", "PASTILLERO"]):
        return "almacenamiento"
    if "BOLSA" in norm:
        return "empaque_flexible"
    return ""

