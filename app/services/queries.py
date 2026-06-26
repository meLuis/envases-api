from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.algorithms import knapsack_budget, knapsack_supply_budget, optimize_purchase_flow
from app.core.graphs import bfs_from_seeds, bfs_path, bidirectional_bfs_path, dijkstra_max_weight_path
from app.core.text import normalize_text, similarity
from app.domain.schemas import QueryResponse
from app.storage.repository import read_csv, read_json


def dataset_summary(dataset_id: str) -> dict:
    return read_json(dataset_id, "dataset_summary.json")


def search_products(dataset_id: str, query: str, limit: int = 10) -> QueryResponse:
    products = read_csv(dataset_id, "products_clean.csv")
    q = normalize_text(query)
    q_terms = set(q.split()) if q else set()

    # Intentar BFS sobre G_attr
    bfs_scores: dict[str, float] = {}
    # product_id → conjunto de etiquetas de atributos normalizadas (para filtro AND)
    product_attr_terms: dict[str, set[str]] = {}
    seeds: list[str] = []
    used_graph = False
    try:
        attr_nodes = read_csv(dataset_id, "semantic_attribute_graph_nodes.csv")
        attr_edges = read_csv(dataset_id, "semantic_attribute_graph_edges.csv")
        seeds = [
            str(row.node_id)
            for row in attr_nodes.itertuples(index=False)
            if row.node_type == "ATTRIBUTE" and q_terms and normalize_text(str(row.label)) in q_terms
        ]
        if seeds:
            reached = bfs_from_seeds(attr_edges, seeds, target_type_prefix="PRODUCT:", max_depth=3)
            for node_id, depth in reached.items():
                product_id = node_id.replace("PRODUCT:", "", 1)
                bfs_scores[product_id] = 0.95 if depth == 1 else 0.70
            used_graph = bool(bfs_scores)

        # Construir mapa product → atributos para el filtro AND multi-término
        if len(q_terms) > 1:
            for edge_row in attr_edges.itertuples(index=False):
                src, tgt = str(edge_row.source), str(edge_row.target)
                if src.startswith("PRODUCT:") and tgt.startswith("ATTR:"):
                    pid = src.replace("PRODUCT:", "", 1)
                    # "ATTR:material:VIDRIO" → tomar todo desde el tercer segmento
                    label = normalize_text(tgt.split(":", 2)[-1]) if tgt.count(":") >= 2 else ""
                    if label:
                        product_attr_terms.setdefault(pid, set()).add(label)
    except FileNotFoundError:
        pass

    rows = []
    for row in products.itertuples(index=False):
        text_score = similarity(q, str(row.product_name_norm))
        if q and q in str(row.product_name_norm):
            text_score = max(text_score, 0.98)
        graph_score = bfs_scores.get(str(row.product_id), 0.0)
        score = max(graph_score, text_score * 0.6) if used_graph else text_score
        if score < 0.35:
            continue

        # Filtro AND: en búsquedas multi-término todos deben estar presentes
        # en los atributos del producto O en su nombre normalizado
        if len(q_terms) > 1:
            pid = str(row.product_id)
            attrs = product_attr_terms.get(pid, set())
            name_words = set(str(row.product_name_norm).split())
            available = attrs | name_words
            if not all(term in available for term in q_terms):
                continue

        rows.append({
            "product_id": row.product_id,
            "product_name": row.product_name,
            "score": round(score, 4),
            "stock": row.stock,
            "match_source": "graph+text" if graph_score > 0 else "text",
        })

    rows = sorted(rows, key=lambda item: item["score"], reverse=True)[:limit]

    algo = (
        "BFS sobre G_attr (prof. 1=atributo exacto, prof. 3=atributo compartido) + filtro AND multi-termino + similitud textual"
        if used_graph
        else "Similitud textual + filtro AND multi-termino"
    )
    return QueryResponse(
        answer=f"Se encontraron {len(rows)} productos compatibles con '{query}'.",
        algorithm=algo,
        table=rows,
        metrics={"matches": len(rows), "limit": limit, "bfs_seeds": len(seeds), "bfs_reached": len(bfs_scores)},
        evidence={"dataset_id": dataset_id, "artifacts": ["products_clean.csv", "semantic_attribute_graph_nodes.csv", "semantic_attribute_graph_edges.csv"], "graph": "G_attr"},
    )


def supplier_risk(dataset_id: str) -> QueryResponse:
    purchases = read_csv(dataset_id, "purchases_clean.csv")
    purchases["quantity"] = pd.to_numeric(purchases["quantity"], errors="coerce").fillna(0)
    purchases["total"] = pd.to_numeric(purchases["total"], errors="coerce").fillna(0)

    # Grado de entrada en G_purchases: cuántos proveedores distintos abastecen cada producto
    per_product = (
        purchases.groupby(["product_id", "product_name"], as_index=False)
        .agg(
            supplier_count=("entity_norm", "nunique"),
            total_units=("quantity", "sum"),
            total_spent=("total", "sum"),
        )
        .sort_values("supplier_count")
    )
    single = per_product[per_product["supplier_count"] == 1].copy()
    # Para los de proveedor único, añadir el nombre del proveedor
    unique_supplier = (
        purchases.groupby("product_id")["entity_name"].first().rename("sole_supplier")
    )
    single = single.merge(unique_supplier, on="product_id", how="left")

    # Concentración por proveedor: % del volumen total que representa cada uno (HHI simplificado)
    supplier_volume = purchases.groupby("entity_name")["quantity"].sum()
    total_volume = supplier_volume.sum()
    supplier_share = (supplier_volume / total_volume * 100).round(2).reset_index()
    supplier_share.columns = ["supplier", "volume_pct"]
    supplier_share = supplier_share.sort_values("volume_pct", ascending=False)

    hhi = round(float(((supplier_volume / total_volume) ** 2).sum() * 10000), 1)

    return QueryResponse(
        answer=(
            f"{len(single)} de {len(per_product)} productos dependen de un unico proveedor. "
            f"Indice de concentracion HHI: {hhi}/10000 "
            f"({'alto riesgo' if hhi > 2500 else 'concentracion moderada' if hhi > 1500 else 'bajo riesgo'})."
        ),
        algorithm="Analisis de grado de entrada en G_purchases + indice HHI",
        table=_records(single),
        metrics={
            "products_analyzed": int(len(per_product)),
            "single_supplier_products": int(len(single)),
            "single_supplier_pct": round(len(single) / len(per_product) * 100, 1) if len(per_product) else 0,
            "hhi": hhi,
            "supplier_concentration": _records(supplier_share.head(10)),
        },
        evidence={"dataset_id": dataset_id, "artifacts": ["purchases_clean.csv"], "graph": "G_purchases"},
    )


def cross_sell(dataset_id: str, product_id: str, limit: int = 10) -> QueryResponse:
    edges = read_csv(dataset_id, "transaction_graph_sales_edges.csv")
    nodes = read_csv(dataset_id, "transaction_graph_sales_nodes.csv")
    pid = str(product_id)
    origin_node = f"PRODUCT:{pid}"

    if origin_node not in edges["source"].values and origin_node not in edges["target"].values:
        return QueryResponse(
            ok=False,
            error=f"Producto {product_id} no tiene ventas registradas en G_sales.",
            algorithm="BFS 2-saltos en G_sales",
        )

    # Construir adjacency del grafo G_sales
    adjacency: dict[str, set[str]] = {}
    for row in edges.itertuples(index=False):
        adjacency.setdefault(str(row.source), set()).add(str(row.target))
        adjacency.setdefault(str(row.target), set()).add(str(row.source))

    # BFS salto 1: PRODUCT:pid → nodos CLIENT vecinos
    client_neighbors = {n for n in adjacency.get(origin_node, set()) if n.startswith("CLIENT:")}
    if not client_neighbors:
        return QueryResponse(
            ok=False,
            error="El producto no tiene clientes asociados en G_sales.",
            algorithm="BFS 2-saltos en G_sales",
        )

    # BFS salto 2: cada CLIENT → otros PRODUCT vecinos (excluir origen)
    co_product_clients: dict[str, set[str]] = {}
    for client_node in client_neighbors:
        for neighbor in adjacency.get(client_node, set()):
            if neighbor.startswith("PRODUCT:") and neighbor != origin_node:
                co_product_clients.setdefault(neighbor, set()).add(client_node)

    if not co_product_clients:
        return QueryResponse(
            ok=False,
            error="Los clientes de este producto no compraron otros productos.",
            algorithm="BFS 2-saltos en G_sales",
        )

    # Construir ranking: productos ordenados por cuántos clientes los comparten
    node_labels = {str(row.node_id): str(row.label) for row in nodes.itertuples(index=False)}
    origin_name = node_labels.get(origin_node, pid)

    ranked = sorted(co_product_clients.items(), key=lambda x: len(x[1]), reverse=True)[:limit]
    table = [
        {
            "product_id": node.replace("PRODUCT:", "", 1),
            "product_name": node_labels.get(node, ""),
            "shared_clients": len(clients),
            "affinity_pct": round(len(clients) / len(client_neighbors) * 100, 1),
        }
        for node, clients in ranked
    ]

    return QueryResponse(
        answer=f"Clientes que compraron '{origin_name}' tambien compraron estos {len(table)} productos.",
        algorithm="BFS 2-saltos en G_sales: PRODUCT -> CLIENT -> PRODUCT",
        table=table,
        metrics={
            "origin_product": pid,
            "buyers_of_origin": len(client_neighbors),
            "co_purchased_products": len(co_product_clients),
            "top_shown": limit,
        },
        evidence={"dataset_id": dataset_id, "artifacts": ["transaction_graph_sales_edges.csv", "transaction_graph_sales_nodes.csv"], "graph": "G_sales"},
    )


def client_supplier_path(dataset_id: str, client: str, supplier: str) -> QueryResponse:
    nodes = read_csv(dataset_id, "transaction_graph_business_nodes.csv")
    edges = read_csv(dataset_id, "transaction_graph_business_edges.csv")
    start = _find_node(nodes, "CLIENT", client)
    goal = _find_node(nodes, "SUPPLIER", supplier)
    if not start or not goal:
        return QueryResponse(ok=False, error="Cliente o proveedor no encontrado en G_business.", algorithm="BFS bidireccional")
    path = bidirectional_bfs_path(edges, start, goal)
    return QueryResponse(
        ok=bool(path),
        answer="Camino encontrado entre cliente y proveedor." if path else "No existe camino en el grafo de negocio.",
        algorithm="BFS bidireccional",
        table=[{"step": index + 1, "node": node, "label": _label(nodes, node)} for index, node in enumerate(path)],
        metrics={"path_length": max(len(path) - 1, 0), "nodes_in_path": len(path)},
        evidence={"dataset_id": dataset_id, "artifacts": ["transaction_graph_business_edges.csv"], "graph": "G_business"},
    )


def product_substitutes(dataset_id: str, product_id: str) -> QueryResponse:
    families = read_csv(dataset_id, "ufds_product_families.csv")
    product_rows = families.loc[families["product_id"].astype(str) == str(product_id)]
    if product_rows.empty:
        return QueryResponse(ok=False, error="Producto no pertenece a una familia UFDS generada.", algorithm="UFDS / componentes conexos")
    family_id = product_rows.iloc[0]["family_id"]
    projection = read_csv(dataset_id, "product_projection_edges.csv")
    projection["similarity"] = pd.to_numeric(projection["similarity"], errors="coerce").fillna(0)
    left = projection.loc[projection["source"].astype(str) == str(product_id)].rename(
        columns={"target": "product_id", "target_name": "product_name"}
    )
    right = projection.loc[projection["target"].astype(str) == str(product_id)].rename(
        columns={"source": "product_id", "source_name": "product_name"}
    )
    related = (
        pd.concat([left, right], ignore_index=True)[["product_id", "product_name", "shared_attributes", "similarity"]]
        .sort_values(["similarity", "shared_attributes"], ascending=False)
        .head(20)
    )
    return QueryResponse(
        answer=f"Producto ubicado en familia {family_id}; se devuelven los vecinos directos mas similares como sustitutos candidatos.",
        algorithm="UFDS / componentes conexos + ranking directo en G_projection",
        table=_records(related.head(20)),
        metrics={"family_id": family_id, "family_size": int(product_rows.iloc[0]["family_size"])},
        evidence={"dataset_id": dataset_id, "artifacts": ["ufds_product_families.csv", "product_projection_edges.csv"], "graph": "G_projection"},
    )


def optimize_budget(dataset_id: str, budget: float, items: list[dict]) -> QueryResponse:
    try:
        options = read_csv(dataset_id, "supply_options.csv")
        for col in ["unit_cost", "capacity_units", "supplier_capacity"]:
            options[col] = pd.to_numeric(options[col], errors="coerce").fillna(0)
        result = knapsack_supply_budget(options, items, budget)
        artifacts = ["supply_options.csv"]
    except FileNotFoundError:
        products = read_csv(dataset_id, "products_clean.csv")
        result = knapsack_budget(products, items, budget)
        artifacts = ["products_clean.csv"]
    return QueryResponse(
        answer=f"Plan calculado con presupuesto {budget}.",
        algorithm="Programacion dinamica - Knapsack",
        table=result["plan"],
        metrics={key: value for key, value in result.items() if key != "plan"},
        evidence={"dataset_id": dataset_id, "artifacts": artifacts},
    )


def best_savings(dataset_id: str, limit: int = 20) -> QueryResponse:
    best = read_csv(dataset_id, "bellman_ford_best_paths.csv")
    return QueryResponse(
        answer="Mejores ahorros historicos frente al costo mediano por producto.",
        algorithm="Bellman-Ford",
        table=_records(best.head(limit)),
        metrics=read_json(dataset_id, "bellman_ford_summary.json"),
        evidence={"dataset_id": dataset_id, "artifacts": ["bellman_ford_best_paths.csv", "bellman_ford_edges.csv"], "graph": "G_offers"},
    )


def optimize_purchase(dataset_id: str, items: list[dict]) -> QueryResponse:
    options = read_csv(dataset_id, "supply_options.csv")
    for col in ["unit_cost", "capacity_units", "supplier_capacity"]:
        options[col] = pd.to_numeric(options[col], errors="coerce").fillna(0)
    order = {str(item["product_id"]): float(item["quantity"]) for item in items}
    result = optimize_purchase_flow(options, order)
    return QueryResponse(
        answer="Pedido optimizado respetando capacidades por proveedor.",
        algorithm="Min-cost flow",
        table=result["plan"],
        metrics={key: value for key, value in result.items() if key != "plan"},
        evidence={"dataset_id": dataset_id, "artifacts": ["supply_options.csv"], "graph": "red de flujo"},
    )


def graph_summary(dataset_id: str) -> dict:
    result: dict[str, Any] = {"dataset_id": dataset_id, "by_graph": {}, "node_type_breakdown": {}, "total_unique_nodes": 0}
    type_counts: dict[str, int] = {}
    for graph_name, metrics_file in [
        ("G_attr", "semantic_attribute_graph_metrics.json"),
        ("G_business", "transaction_graph_business_metrics.json"),
        ("G_sales", "transaction_graph_sales_metrics.json"),
        ("G_purchases", "transaction_graph_purchases_metrics.json"),
    ]:
        try:
            m = read_json(dataset_id, metrics_file)
            result["by_graph"][graph_name] = {"nodes": m.get("node_count", 0), "edges": m.get("edge_count", 0)}
            for node_type, count in m.get("node_type_counts", {}).items():
                type_counts[node_type] = max(type_counts.get(node_type, 0), count)
        except FileNotFoundError:
            result["by_graph"][graph_name] = {"nodes": 0, "edges": 0, "available": False}
    try:
        proj = read_json(dataset_id, "product_projection_metrics.json")
        result["by_graph"]["G_projection"] = {"nodes": proj.get("product_count", 0), "edges": proj.get("edge_count", 0)}
    except FileNotFoundError:
        pass
    result["node_type_breakdown"] = type_counts
    result["total_unique_nodes"] = sum(type_counts.values())
    return result


def weighted_connection(dataset_id: str, source: str, target: str) -> QueryResponse:
    nodes = read_csv(dataset_id, "transaction_graph_business_nodes.csv")
    edges = read_csv(dataset_id, "transaction_graph_business_edges.csv")
    edges["amount"] = pd.to_numeric(edges["amount"], errors="coerce").fillna(0)
    source_node = _find_any_node(nodes, source)
    target_node = _find_any_node(nodes, target)
    if not source_node or not target_node:
        return QueryResponse(ok=False, error="No se encontro origen o destino.", algorithm="Dijkstra ponderado")
    path, total_weight = dijkstra_max_weight_path(edges, source_node, target_node, weight_col="amount")
    return QueryResponse(
        ok=bool(path),
        answer=(
            f"Camino comercialmente mas significativo encontrado (S/{total_weight} en transacciones)."
            if path else "No hay conexion entre las entidades."
        ),
        algorithm="Dijkstra ponderado por volumen comercial (amount) en G_business",
        table=[{"step": i + 1, "node": node, "label": _label(nodes, node)} for i, node in enumerate(path)],
        metrics={"path_length": max(len(path) - 1, 0), "total_commercial_volume": total_weight},
        evidence={"dataset_id": dataset_id, "artifacts": ["transaction_graph_business_edges.csv"], "graph": "G_business"},
    )


def _find_node(nodes: pd.DataFrame, node_type: str, text: str) -> str | None:
    key = normalize_text(text)
    subset = nodes.loc[nodes["node_type"] == node_type]
    for row in subset.itertuples(index=False):
        if key in str(row.node_id) or key in normalize_text(row.label):
            return str(row.node_id)
    return None


def _find_any_node(nodes: pd.DataFrame, text: str) -> str | None:
    key = normalize_text(text)
    for row in nodes.itertuples(index=False):
        if key in str(row.node_id) or key in normalize_text(row.label):
            return str(row.node_id)
    return None


def _label(nodes: pd.DataFrame, node_id: str) -> str:
    row = nodes.loc[nodes["node_id"] == node_id]
    return "" if row.empty else str(row.iloc[0]["label"])


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.fillna("").to_dict("records")
