from __future__ import annotations

from collections import defaultdict, deque
import heapq
from itertools import combinations
import json
from typing import Any

import pandas as pd

from app.config import MIN_SUPPLIER_PROJECTION_SIMILARITY, MIN_SUPPLIER_SHARED_PRODUCTS
from app.core.text import normalize_text


# Capas semánticas de G_attr: (columna en product_attributes, node_type, relación).
# Cada valor distinto de una capa es un nodo propio (TYPE:FRASCO, COLOR:AMBAR…).
ATTRIBUTE_SPECS = [
    ("product_type", "TYPE", "HAS_TYPE"),
    ("subtype", "SUBTYPE", "HAS_SUBTYPE"),
    ("accessory", "ACCESSORY", "HAS_ACCESSORY"),
    ("shape", "SHAPE", "HAS_SHAPE"),
    ("feature", "FEATURE", "HAS_FEATURE"),
    ("material", "MATERIAL", "HAS_MATERIAL"),
    ("color", "COLOR", "HAS_COLOR"),
    ("capacity_text", "CAPACITY", "HAS_CAPACITY"),
    ("mouth_size_text", "MOUTH_SIZE", "HAS_MOUTH_SIZE"),
]

# Clave dentro de attribute_confidence (JSON por producto) para cada columna.
CONFIDENCE_KEYS = {
    "product_type": "product_type",
    "subtype": "subtype",
    "accessory": "accessory",
    "shape": "shape",
    "feature": "feature",
    "material": "material",
    "color": "color",
    "capacity_text": "capacity",
    "mouth_size_text": "mouth_size",
}

MIN_ATTRIBUTE_CONFIDENCE = 0.75


def _stable(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def attr_node_id(node_type: str, value: object) -> str:
    """Id de nodo de atributo: TYPE:FRASCO, COLOR:AMBAR, CAPACITY:100ML."""
    norm = normalize_text(value).replace(" ", "_")
    return f"{node_type}:{norm}"


def _split_values(value: object) -> list[str]:
    text = _stable(value)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _parse_confidence(value: object) -> dict[str, float]:
    text = _stable(value)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}


def _prepare_attributes(attributes: pd.DataFrame) -> pd.DataFrame:
    """Deriva capacity_text ('100ML') y mouth_size_text ('18MM') para las capas."""
    prepared = attributes.copy()

    def cap_text(row: pd.Series) -> str:
        value = row.get("capacity_value")
        unit = row.get("capacity_unit")
        if _stable(value) and _stable(unit):
            try:
                return f"{float(value):g}{str(unit).upper()}"
            except (TypeError, ValueError):
                return ""
        return ""

    def mouth_text(row: pd.Series) -> str:
        value = row.get("mouth_size_mm")
        if _stable(value):
            try:
                return f"{float(value):g}MM"
            except (TypeError, ValueError):
                return ""
        return ""

    prepared["capacity_text"] = prepared.apply(cap_text, axis=1) if not prepared.empty else ""
    prepared["mouth_size_text"] = prepared.apply(mouth_text, axis=1) if not prepared.empty else ""
    return prepared


def _sold_units_by_product(activity: pd.DataFrame | None) -> dict[str, float]:
    """product_id -> unidades vendidas, desde el resumen de actividad de ventas.

    Se usa para anotar cada nodo-producto de G_attr con su popularidad, de modo
    que el buscador y el grafo interactivo ordenen finalistas por ventas sin
    consultar nada extra.
    """
    if activity is None or activity.empty or "product_id" not in activity.columns:
        return {}
    col = "sold_units" if "sold_units" in activity.columns else None
    if col is None:
        return {}
    sold: dict[str, float] = {}
    for row in activity.itertuples(index=False):
        pid = _stable(getattr(row, "product_id", ""))
        if not pid:
            continue
        try:
            sold[pid] = float(getattr(row, col, 0) or 0)
        except (TypeError, ValueError):
            sold[pid] = 0.0
    return sold


def build_semantic_graph(
    attributes: pd.DataFrame, activity: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """G_attr en capas: PRODUCT ↔ {TYPE, SUBTYPE, …, CAPACITY, MOUTH_SIZE}.

    Cada valor distinto de una capa es un nodo (TYPE:FRASCO, COLOR:AMBAR…).
    El peso de la arista es la confianza de extracción de ESE atributo; el
    filtro min_confidence se aplica por atributo (un atributo débil no descarta
    los fuertes del mismo producto).

    `activity` (opcional, columnas product_id/sold_units) anota cada nodo-producto
    con `units_sold` para ordenar finalistas por popularidad.
    """
    data = _prepare_attributes(attributes)
    sold_units = _sold_units_by_product(activity)
    nodes: dict[str, dict[str, Any]] = {}
    edges = []
    for row in data.itertuples(index=False):
        product_id = _stable(getattr(row, "product_id", ""))
        if not product_id:
            continue
        product_node = f"PRODUCT:{product_id}"
        nodes[product_node] = {
            "node_id": product_node,
            "node_type": "PRODUCT",
            "label": _stable(getattr(row, "product_name", "")),
            "ref": product_id,
            "units_sold": sold_units.get(product_id, 0.0),
        }

        product_conf = 0.0
        try:
            product_conf = float(getattr(row, "confidence", 0) or 0)
        except (TypeError, ValueError):
            product_conf = 0.0
        confidences = _parse_confidence(getattr(row, "attribute_confidence", ""))

        for column, attr_type, relation in ATTRIBUTE_SPECS:
            attr_conf = confidences.get(CONFIDENCE_KEYS[column], product_conf)
            if attr_conf < MIN_ATTRIBUTE_CONFIDENCE:
                continue
            for value in _split_values(getattr(row, column, "")):
                attr_node = attr_node_id(attr_type, value)
                nodes.setdefault(
                    attr_node,
                    {
                        "node_id": attr_node,
                        "node_type": attr_type,
                        "label": value,
                        "ref": column,
                    },
                )
                edges.append(
                    {
                        "source": product_node,
                        "target": attr_node,
                        "edge_type": relation,
                        "weight": round(attr_conf, 4),
                    }
                )
    node_df = pd.DataFrame(nodes.values())
    edge_df = pd.DataFrame(edges)
    return node_df, edge_df, _metrics(node_df, edge_df, "G_attr")


def build_supplier_projection(purchases: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Similitud proveedor-proveedor: Jaccard sobre el catalogo de productos que cada uno vende.

    La similitud es un criterio operativo: cuanto catalogo real de compras
    comparten dos proveedores. Sirve para "si este proveedor falla, quien mas
    cubre lo mismo".
    """
    if purchases.empty:
        return pd.DataFrame(), {"graph_name": "G_supplier_projection", "edge_count": 0, "supplier_count": 0}

    supplier_products: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for entity_norm, group in purchases.groupby("entity_norm"):
        supplier_id = _stable(entity_norm)
        if not supplier_id:
            continue
        supplier_products[supplier_id] = {
            _stable(product_id) for product_id in group["product_id"] if _stable(product_id)
        }
        names[supplier_id] = _stable(group["entity_name"].iloc[0])

    rows = []
    for left, right in combinations(supplier_products, 2):
        union = supplier_products[left] | supplier_products[right]
        if not union:
            continue
        shared = supplier_products[left] & supplier_products[right]
        score = len(shared) / len(union)
        if score >= MIN_SUPPLIER_PROJECTION_SIMILARITY and len(shared) >= MIN_SUPPLIER_SHARED_PRODUCTS:
            left_count = len(supplier_products[left])
            right_count = len(supplier_products[right])
            rows.append(
                {
                    "source": left,
                    "target": right,
                    "source_name": names[left],
                    "target_name": names[right],
                    "shared_products": len(shared),
                    "source_product_count": left_count,
                    "target_product_count": right_count,
                    "source_coverage": round(len(shared) / left_count, 4) if left_count else 0,
                    "target_coverage": round(len(shared) / right_count, 4) if right_count else 0,
                    "similarity": round(score, 4),
                }
            )
    frame = pd.DataFrame(rows).sort_values("similarity", ascending=False) if rows else pd.DataFrame()
    return frame, {
        "graph_name": "G_supplier_projection",
        "edge_count": int(len(frame)),
        "supplier_count": len(supplier_products),
    }


def build_transaction_graphs(
    sales: pd.DataFrame, purchases: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]]:
    return {
        "sales": _transaction_graph(sales, "CLIENT", "SALE"),
        "purchases": _transaction_graph(purchases, "SUPPLIER", "PURCHASE"),
        "business": _business_graph(sales, purchases),
    }


def dijkstra_max_weight_path(edges: pd.DataFrame, start: str, goal: str, weight_col: str = "amount") -> tuple[list[str], float]:
    """Dijkstra invertido: encuentra el camino que maximiza el peso acumulado.

    Usa 1/(w+1) como costo para que el camino de menor costo sea el de mayor
    volumen comercial. Devuelve (path, total_weight).
    """
    weights: dict[tuple[str, str], float] = {}
    for row in edges.itertuples(index=False):
        src, tgt = str(row.source), str(row.target)
        w = float(getattr(row, weight_col, 0) or 0)
        weights[(src, tgt)] = max(weights.get((src, tgt), 0), w)
        weights[(tgt, src)] = max(weights.get((tgt, src), 0), w)

    adjacency = _adjacency(edges)
    if start not in adjacency or goal not in adjacency:
        return [], 0.0

    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, str | None] = {start: None}
    heap = [(0.0, start)]

    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            break
        if cost > dist.get(node, float("inf")):
            continue
        for neighbor in adjacency.get(node, set()):
            w = weights.get((node, neighbor), 0.0)
            edge_cost = 1.0 / (w + 1.0)
            new_cost = cost + edge_cost
            if new_cost < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_cost
                prev[neighbor] = node
                heapq.heappush(heap, (new_cost, neighbor))

    if goal not in prev and goal != start:
        return [], 0.0

    path = []
    node: str | None = goal
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    total_weight = sum(weights.get((path[i], path[i + 1]), 0.0) for i in range(len(path) - 1))
    return path, round(total_weight, 2)


def bfs_path(edges: pd.DataFrame, start: str, goal: str) -> list[str]:
    adjacency = _adjacency(edges)
    if start not in adjacency or goal not in adjacency:
        return []
    queue = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(path + [neighbor])
    return []


def bidirectional_bfs_path(edges: pd.DataFrame, start: str, goal: str) -> list[str]:
    adjacency = _adjacency(edges)
    if start not in adjacency or goal not in adjacency:
        return []
    if start == goal:
        return [start]
    front = {start}
    back = {goal}
    parents_front = {start: None}
    parents_back = {goal: None}
    while front and back:
        if len(front) <= len(back):
            meet = _expand(front, adjacency, parents_front, parents_back)
        else:
            meet = _expand(back, adjacency, parents_back, parents_front)
        if meet:
            return _reconstruct(meet, parents_front, parents_back)
    return []


def _transaction_graph(frame: pd.DataFrame, entity_type: str, graph_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    nodes: dict[str, dict[str, Any]] = {}
    edges = []
    for row in frame.itertuples(index=False):
        entity_node = f"{entity_type}:{row.entity_norm}"
        product_node = f"PRODUCT:{row.product_id}"
        document = str(getattr(row, "document", "") or "").strip()
        nodes[entity_node] = {"node_id": entity_node, "node_type": entity_type, "label": row.entity_name, "ref": row.entity_norm}
        nodes[product_node] = {"node_id": product_node, "node_type": "PRODUCT", "label": row.product_name, "ref": row.product_id}
        edges.append(
            {
                "source": entity_node,
                "target": product_node,
                "edge_type": graph_name.lower(),
                "weight": float(row.quantity or 0),
                "amount": float(row.total or 0),
            }
        )
        if document:
            document_node = f"{graph_name}_DOC:{document}"
            nodes[document_node] = {
                "node_id": document_node,
                "node_type": "DOCUMENT",
                "label": document,
                "ref": document,
            }
            for source, target, relation in (
                (entity_node, document_node, f"{graph_name.lower()}_document"),
                (document_node, product_node, f"{graph_name.lower()}_line"),
            ):
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "edge_type": relation,
                        "weight": float(row.quantity or 0),
                        "amount": float(row.total or 0),
                    }
                )
    node_df = pd.DataFrame(nodes.values())
    edge_df = pd.DataFrame(edges)
    return node_df, edge_df, _metrics(node_df, edge_df, f"G_{graph_name.lower()}")


def _business_graph(sales: pd.DataFrame, purchases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sales_nodes, sales_edges, _ = _transaction_graph(sales, "CLIENT", "SALE")
    purchase_nodes, purchase_edges, _ = _transaction_graph(purchases, "SUPPLIER", "PURCHASE")
    nodes = pd.concat([sales_nodes, purchase_nodes], ignore_index=True).drop_duplicates("node_id")
    edges = pd.concat([sales_edges, purchase_edges], ignore_index=True)
    return nodes, edges, _metrics(nodes, edges, "G_business")


def _unique_edge_count(edges: pd.DataFrame) -> int:
    # El grafo real (visualizacion + BFS/Dijkstra/_adjacency) es simple y no
    # dirigido: pares repetidos (mismo cliente/producto en varias facturas)
    # colapsan a una sola arista. Contar filas crudas del CSV de transacciones
    # infla el numero frente a lo que realmente se recorre y se dibuja.
    if edges.empty:
        return 0
    pairs = {
        (s, t) if s <= t else (t, s)
        for s, t in zip(edges["source"].astype(str), edges["target"].astype(str))
    }
    return len(pairs)


def _metrics(nodes: pd.DataFrame, edges: pd.DataFrame, name: str) -> dict:
    return {
        "graph_name": name,
        "node_count": int(len(nodes)),
        "edge_count": _unique_edge_count(edges),
        "node_type_counts": nodes["node_type"].value_counts().to_dict() if not nodes.empty else {},
    }


def _adjacency(edges: pd.DataFrame) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    if edges.empty:
        return graph
    for row in edges.itertuples(index=False):
        graph[str(row.source)].add(str(row.target))
        graph[str(row.target)].add(str(row.source))
    return graph


def _expand(frontier: set[str], adjacency: dict[str, set[str]], parents_this: dict, parents_other: dict) -> str | None:
    next_frontier = set()
    for node in list(frontier):
        for neighbor in adjacency[node]:
            if neighbor in parents_this:
                continue
            parents_this[neighbor] = node
            if neighbor in parents_other:
                return neighbor
            next_frontier.add(neighbor)
    frontier.clear()
    frontier.update(next_frontier)
    return None


def _reconstruct(meet: str, parents_front: dict, parents_back: dict) -> list[str]:
    left = []
    node = meet
    while node is not None:
        left.append(node)
        node = parents_front[node]
    left.reverse()
    right = []
    node = parents_back[meet]
    while node is not None:
        right.append(node)
        node = parents_back[node]
    return left + right
