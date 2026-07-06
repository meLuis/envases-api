from __future__ import annotations

from datetime import datetime
import re
from typing import Any

import pandas as pd

from app.core.geo import parse_lat_lon
from app.core.text import normalize_column, normalize_text, similarity


ALIASES: dict[str, dict[str, list[str]]] = {
    "products": {
        "product_id": ["codigo interno", "codigo", "cod producto", "sku"],
        "product_name": ["descripcion", "producto", "nombre", "articulo"],
        "stock": ["stock actual disponible", "stock", "existencia"],
        "category": ["descripcion de categoria", "categoria"],
        "unit": ["codigo unidad de medida", "unidad de medida", "unidad"],
        # Precios: compramos con IGV, vendemos con precio listado sin IGV
        "unit_cost_purchase": ["precio compra unitario con igv", "precio compra unitario"],
        "unit_price_sale": ["valor venta unitario sin igv", "valor venta unitario"],
    },
    "sales": {
        "date": ["fecha de emision", "fecha", "fec emision"],
        "doc_type": ["tipo"],
        "doc_serie": ["serie"],
        "document": ["numero", "número"],
        "customer_id": ["doc entidad numero", "ruc cliente", "documento cliente"],
        "customer": ["denominacion entidad", "cliente", "razon social"],
        "product_id": ["codigo", "cod producto", "sku"],
        "product_name": ["descripcion", "producto", "articulo"],
        "quantity": ["cantidad", "cant"],
        "unit_price": ["precio unitario", "valor unitario"],
        "subtotal": ["subtotal"],
        "total": ["total"],
        "voided": ["anulado"],
        # Coordenadas del cliente (solo en el dataset sintético/logístico; opcional).
        "lat": ["latitud", "lat", "latitud cliente"],
        "lon": ["longitud", "lon", "lng", "longitud cliente"],
    },
    "purchases": {
        "date": ["fecha de emision", "fecha", "fec emision"],
        "doc_type": ["tipo"],
        "doc_serie": ["serie"],
        "document": ["numero", "número"],
        "supplier_id": ["doc entidad numero", "ruc proveedor", "documento proveedor"],
        "supplier": ["denominacion entidad", "proveedor", "razon social"],
        "product_id": ["codigo", "cod producto", "sku"],
        "product_name": ["descripcion", "producto", "articulo"],
        "quantity": ["cantidad", "cant"],
        "unit_cost": ["precio unitario", "valor unitario", "costo unitario"],
        "subtotal": ["subtotal"],
        "total": ["total"],
        # Coordenadas del proveedor (solo en el dataset sintético/logístico; opcional).
        "lat": ["latitud", "lat", "latitud proveedor"],
        "lon": ["longitud", "lon", "lng", "longitud proveedor"],
    },
}


def detect_schema(frame: pd.DataFrame, kind: str) -> dict[str, dict[str, Any]]:
    columns = list(frame.columns)
    normalized = {col: normalize_column(col) for col in columns}
    mapping: dict[str, dict[str, Any]] = {}
    for field, aliases in ALIASES[kind].items():
        best_col = None
        best_score = 0.0
        for col, norm_col in normalized.items():
            for alias in aliases:
                score = 1.0 if normalize_column(alias) == norm_col else similarity(alias, col)
                if score > best_score:
                    best_col = col
                    best_score = score
        mapping[field] = {
            "column": best_col if best_score >= 0.62 else None,
            "confidence": round(best_score, 4),
        }
    return mapping


def normalize_all(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    mappings = {kind: detect_schema(frame, kind) for kind, frame in raw.items()}
    cleaned = {
        "products": normalize_products(raw["products"], mappings["products"]),
        "sales": normalize_transactions(raw["sales"], mappings["sales"], "sale"),
        "purchases": normalize_transactions(raw["purchases"], mappings["purchases"], "purchase"),
    }
    return cleaned, mappings


def normalize_products(frame: pd.DataFrame, mapping: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        product_id = _value(row, mapping, "product_id")
        product_name = _value(row, mapping, "product_name")
        if not product_id and not product_name:
            continue
        rows.append(
            {
                "product_id": str(product_id).strip(),
                "product_name": _clean_description(product_name),
                "product_name_norm": normalize_text(product_name),
                "stock": _to_float(_value(row, mapping, "stock")),
                "category": str(_value(row, mapping, "category") or "").strip(),
                "unit": str(_value(row, mapping, "unit") or "").strip(),
                "unit_cost_purchase": _to_float(_value(row, mapping, "unit_cost_purchase")),
                "unit_price_sale": _to_float(_value(row, mapping, "unit_price_sale")),
            }
        )
    result = pd.DataFrame(rows).drop_duplicates("product_id")
    return result.sort_values("product_id") if not result.empty else result


def normalize_transactions(frame: pd.DataFrame, mapping: dict[str, Any], kind: str) -> pd.DataFrame:
    entity_id_field = "customer_id" if kind == "sale" else "supplier_id"
    entity_field = "customer" if kind == "sale" else "supplier"
    price_field = "unit_price" if kind == "sale" else "unit_cost"
    rows = []
    for _, row in frame.iterrows():
        product_id = str(_value(row, mapping, "product_id") or "").strip()
        product_name = _clean_description(_value(row, mapping, "product_name"))
        entity = str(_value(row, mapping, entity_field) or "").strip()
        if not product_id and not product_name:
            continue
        if kind == "sale" and normalize_text(_value(row, mapping, "voided")) in {"SI", "YES", "ANULADO"}:
            continue
        quantity = _to_float(_value(row, mapping, "quantity"))
        unit_value = _to_float(_value(row, mapping, price_field))
        total = _to_float(_value(row, mapping, "total"))
        subtotal = _to_float(_value(row, mapping, "subtotal"))
        if unit_value <= 0 and quantity > 0:
            unit_value = (subtotal or total) / quantity if (subtotal or total) else 0.0
        coord = parse_lat_lon(_value(row, mapping, "lat"), _value(row, mapping, "lon"))
        rows.append(
            {
                "date": _parse_date(_value(row, mapping, "date")),
                "document": _compound_document(
                    _value(row, mapping, "doc_type"),
                    _value(row, mapping, "doc_serie"),
                    _value(row, mapping, "document"),
                ),
                "entity_id": str(_value(row, mapping, entity_id_field) or "").strip(),
                "entity_name": entity or "SIN ENTIDAD",
                "entity_norm": normalize_text(entity or "SIN ENTIDAD"),
                "product_id": product_id,
                "product_name": product_name,
                "product_name_norm": normalize_text(product_name),
                "quantity": quantity,
                "unit_value": unit_value,
                "subtotal": subtotal,
                "total": total,
                "lat": coord[0] if coord else "",
                "lon": coord[1] if coord else "",
                "transaction_type": kind,
            }
        )
    result = pd.DataFrame(rows)
    return result if result.empty else result.sort_values(["date", "product_id"])


def entity_coordinates(cleaned: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Coordenadas por nodo de negocio (CLIENT:… / SUPPLIER:…) para el A* logístico.

    Toma la primera coordenada válida de cada entidad en ventas (clientes) y
    compras (proveedores). Devuelve {} si el dataset no trae lat/lon (dataset
    real): en ese caso A* responde con el mensaje "requiere dataset sintético".
    """
    coords: dict[str, tuple[float, float]] = {}
    for kind, prefix in (("sales", "CLIENT"), ("purchases", "SUPPLIER")):
        frame = cleaned.get(kind)
        if frame is None or frame.empty or "lat" not in frame.columns:
            continue
        for row in frame.itertuples(index=False):
            entity_norm = str(getattr(row, "entity_norm", "") or "")
            if not entity_norm:
                continue
            node_id = f"{prefix}:{entity_norm}"
            if node_id in coords:
                continue
            coord = parse_lat_lon(getattr(row, "lat", ""), getattr(row, "lon", ""))
            if coord:
                coords[node_id] = coord
    return coords


def profile_frames(raw: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        kind: {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "empty_cells": int(frame.isna().sum().sum()),
        }
        for kind, frame in raw.items()
    }


def quality_summary(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for kind, frame in cleaned.items():
        rows.append(
            {
                "dataset": kind,
                "rows": int(len(frame)),
                "products": int(frame["product_id"].nunique()) if "product_id" in frame else 0,
                "entities": int(frame["entity_norm"].nunique()) if "entity_norm" in frame else 0,
                "missing_product_id": int((frame.get("product_id", pd.Series(dtype=str)) == "").sum()),
            }
        )
    return pd.DataFrame(rows)


def _value(row: pd.Series, mapping: dict[str, Any], field: str) -> Any:
    col = mapping.get(field, {}).get("column")
    return row.get(col) if col else None


def _to_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    text = text.replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date().isoformat()
        except ValueError:
            continue
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _compound_document(doc_type: Any, doc_serie: Any, doc_number: Any) -> str:
    """Construye identificador de documento compuesto: TIPO-SERIE-NUMERO.

    Si solo existe un campo (otra empresa con columna única de documento),
    lo usa directamente. El resultado es único por orden de compra/venta.
    """
    parts = [str(v).strip() for v in (doc_type, doc_serie, doc_number) if v and str(v).strip() not in ("", "None", "nan", "-")]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "-".join(parts)


def _clean_description(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    text = re.sub(r"\s+", " ", text)
    midpoint = len(text) // 2
    if midpoint > 8 and text[:midpoint].strip() == text[midpoint:].strip():
        return text[:midpoint].strip()
    return text

