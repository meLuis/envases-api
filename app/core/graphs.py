from __future__ import annotations

from collections import defaultdict, deque
import heapq
from itertools import combinations
from typing import Any

import pandas as pd


def build_semantic_graph(attributes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    nodes: dict[str, dict[str, Any]] = {}
    edges = []
    attr_cols = ["product_type", "material", "color", "capacity", "mouth", "features", "category_use"]
    for row in attributes.itertuples(index=False):
        product_node = f"PRODUCT:{row.product_id}"
        nodes[product_node] = {
            "node_id": product_node,
            "node_type": "PRODUCT",
            "label": row.product_name,
            "ref": row.product_id,
        }
        for col in attr_cols:
            raw = getattr(row, col)
            values = str(raw).split("|") if col == "features" and raw else [raw]
            for value in values:
                value = str(value or "").strip()
                if not value:
                    continue
                attr_node = f"ATTR:{col}:{value}"
                nodes[attr_node] = {
                    "node_id": attr_node,
                    "node_type": "ATTRIBUTE",
                    "label": value,
                    "ref": col,
                }
                edges.append(
                    {
                        "source": product_node,
                        "target": attr_node,
                        "edge_type": col,
                        "weight": 1.0,
                    }
                )
    node_df = pd.DataFrame(nodes.values())
    edge_df = pd.DataFrame(edges)
    return node_df, edge_df, _metrics(node_df, edge_df, "G_attr")


def build_product_projection(attributes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    attr_cols = ["product_type", "material", "color", "capacity", "mouth", "features", "category_use"]
    product_attrs: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for row in attributes.itertuples(index=False):
        values = set()
        for col in attr_cols:
            raw = getattr(row, col)
            for value in str(raw or "").split("|"):
                value = value.strip()
                if value:
                    values.add(f"{col}:{value}")
        product_attrs[str(row.product_id)] = values
        names[str(row.product_id)] = str(row.product_name)
    rows = []
    for left, right in combinations(product_attrs, 2):
        union = product_attrs[left] | product_attrs[right]
        if not union:
            continue
        shared = product_attrs[left] & product_attrs[right]
        score = len(shared) / len(union)
        if score >= 0.28:
            rows.append(
                {
                    "source": left,
                    "target": right,
                    "source_name": names[left],
                    "target_name": names[right],
                    "shared_attributes": len(shared),
                    "similarity": round(score, 4),
                }
            )
    frame = pd.DataFrame(rows).sort_values("similarity", ascending=False) if rows else pd.DataFrame()
    return frame, {"graph_name": "G_projection", "edge_count": int(len(frame)), "product_count": len(product_attrs)}


def build_transaction_graphs(
    sales: pd.DataFrame, purchases: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]]:
    return {
        "sales": _transaction_graph(sales, "CLIENT", "SALE"),
        "purchases": _transaction_graph(purchases, "SUPPLIER", "PURCHASE"),
        "business": _business_graph(sales, purchases),
    }


def bfs_from_seeds(
    edges: pd.DataFrame,
    seed_nodes: list[str],
    target_type_prefix: str,
    max_depth: int = 2,
) -> dict[str, int]:
    """BFS desde nodos semilla; devuelve {node_id: profundidad} para nodos del tipo buscado."""
    adjacency = _adjacency(edges)
    result: dict[str, int] = {}
    visited = set(seed_nodes)
    queue: deque[tuple[str, int]] = deque((node, 0) for node in seed_nodes)
    while queue:
        node, depth = queue.popleft()
        if node.startswith(target_type_prefix) and node not in seed_nodes:
            result[node] = depth
        if depth >= max_depth:
            continue
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return result


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
    node_df = pd.DataFrame(nodes.values())
    edge_df = pd.DataFrame(edges)
    return node_df, edge_df, _metrics(node_df, edge_df, f"G_{graph_name.lower()}")


def _business_graph(sales: pd.DataFrame, purchases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sales_nodes, sales_edges, _ = _transaction_graph(sales, "CLIENT", "SALE")
    purchase_nodes, purchase_edges, _ = _transaction_graph(purchases, "SUPPLIER", "PURCHASE")
    nodes = pd.concat([sales_nodes, purchase_nodes], ignore_index=True).drop_duplicates("node_id")
    edges = pd.concat([sales_edges, purchase_edges], ignore_index=True)
    return nodes, edges, _metrics(nodes, edges, "G_business")


def _metrics(nodes: pd.DataFrame, edges: pd.DataFrame, name: str) -> dict:
    return {
        "graph_name": name,
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
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

