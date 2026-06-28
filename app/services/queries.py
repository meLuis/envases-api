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


# ============================================================================
# ANALISIS DOCUMENTALES: 5 algoritmos sobre nodos DOCUMENT en grafos transaccionales
# ============================================================================


def product_co_occurrence(dataset_id: str, product_id: str, graph_type: str = "sales", limit: int = 15) -> QueryResponse:
    """
    RECOMENDACIÓN: Market Basket Analysis.

    Para un producto, encuentra qué otros productos aparecen en el MISMO DOCUMENTO.
    Diferencia con cross_sell:
    - cross_sell mide afinidad histórica (clientes que compraron A también compraron B en cualquier momento)
    - co_occurrence mide co-compra operativa (B estaba en la misma factura/comprobante que A)

    Usa: aristas DOCUMENT -> PRODUCT (edge_type: sale_line o purchase_line)
    """
    if graph_type not in ("sales", "purchases"):
        graph_type = "sales"

    try:
        edges = read_csv(dataset_id, f"transaction_graph_{graph_type}_edges.csv")
    except FileNotFoundError:
        return QueryResponse(ok=False, error=f"Grafo G_{graph_type} no disponible.", algorithm="Market Basket Analysis")

    edges["amount"] = pd.to_numeric(edges["amount"], errors="coerce").fillna(0)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0)

    pid = str(product_id)
    origin_node = f"PRODUCT:{pid}"

    # Encontrar todos los documentos que contienen este producto
    doc_edges = edges[edges[f"edge_type"].str.contains("line", na=False)]
    documents_with_origin = doc_edges[doc_edges["target"] == origin_node]["source"].unique()

    if len(documents_with_origin) == 0:
        return QueryResponse(
            ok=False,
            error=f"Producto {product_id} no aparece en documentos de {graph_type}.",
            algorithm="Market Basket Analysis - BFS en DOCUMENT nodes"
        )

    # Por cada documento, listar otros productos
    co_products: dict[str, dict] = {}
    for doc_node in documents_with_origin:
        # Encontrar todos los productos en este documento
        products_in_doc = doc_edges[doc_edges["source"] == doc_node]
        for row in products_in_doc.itertuples(index=False):
            if row.target != origin_node:
                prod_id = row.target.replace("PRODUCT:", "", 1)
                if prod_id not in co_products:
                    co_products[prod_id] = {"frequency": 0, "total_quantity": 0.0, "total_amount": 0.0, "documents": []}
                co_products[prod_id]["frequency"] += 1
                co_products[prod_id]["total_quantity"] += float(row.weight or 0)
                co_products[prod_id]["total_amount"] += float(row.amount or 0)
                co_products[prod_id]["documents"].append(doc_node)

    if not co_products:
        return QueryResponse(
            ok=False,
            error="El producto no aparece con otros en documentos.",
            algorithm="Market Basket Analysis - BFS en DOCUMENT nodes"
        )

    # Obtener nombres de productos
    nodes = read_csv(dataset_id, f"transaction_graph_{graph_type}_nodes.csv")
    node_labels = {str(row.node_id): str(row.label) for row in nodes.itertuples(index=False)}
    origin_name = node_labels.get(origin_node, pid)

    # Ranking por frecuencia y volumen
    ranked = sorted(
        co_products.items(),
        key=lambda x: (x[1]["frequency"], x[1]["total_amount"]),
        reverse=True
    )[:limit]

    table = [
        {
            "product_id": prod_id,
            "product_name": node_labels.get(f"PRODUCT:{prod_id}", ""),
            "co_occurrences": co_products[prod_id]["frequency"],
            "total_quantity": round(co_products[prod_id]["total_quantity"], 2),
            "total_amount": round(co_products[prod_id]["total_amount"], 2),
            "avg_amount_per_doc": round(co_products[prod_id]["total_amount"] / co_products[prod_id]["frequency"], 2),
        }
        for prod_id, data in ranked
    ]

    return QueryResponse(
        answer=f"'{origin_name}' aparece en {len(documents_with_origin)} documentos. Se encontraron {len(co_products)} productos co-comprados en la misma operación.",
        algorithm="Market Basket Analysis - BFS bidireccional en DOCUMENT nodes",
        table=table,
        metrics={
            "origin_product": pid,
            "documents_containing_origin": int(len(documents_with_origin)),
            "unique_co_products": len(co_products),
            "top_shown": limit,
        },
        evidence={
            "dataset_id": dataset_id,
            "artifacts": [f"transaction_graph_{graph_type}_edges.csv", f"transaction_graph_{graph_type}_nodes.csv"],
            "graph": f"G_{graph_type}",
            "model": "DOCUMENT node as transaction context"
        },
    )


def product_volatility(dataset_id: str, product_id: str, graph_type: str = "sales") -> QueryResponse:
    """
    PUNTO 2: Volatilidad de co-compra.

    ¿Un producto siempre aparece con los mismos otros productos, o varía según el documento?
    - Alta volatilidad: aparece con diferentes combinaciones → "versátil"
    - Baja volatilidad: aparece con conjunto fijo → "dependiente"

    Métrica: Jaccard similarity promedio entre documentos.
    """
    if graph_type not in ("sales", "purchases"):
        graph_type = "sales"

    try:
        edges = read_csv(dataset_id, f"transaction_graph_{graph_type}_edges.csv")
    except FileNotFoundError:
        return QueryResponse(ok=False, error=f"Grafo G_{graph_type} no disponible.", algorithm="Volatility Analysis")

    pid = str(product_id)
    origin_node = f"PRODUCT:{pid}"
    doc_edges = edges[edges["edge_type"].str.contains("line", na=False)]
    documents_with_origin = doc_edges[doc_edges["target"] == origin_node]["source"].unique()

    if len(documents_with_origin) < 2:
        return QueryResponse(
            ok=False,
            error="Se necesitan al menos 2 documentos para calcular volatilidad.",
            algorithm="Volatility Analysis"
        )

    # Construir lista de productos por documento
    doc_products: dict[str, set[str]] = {}
    for doc_node in documents_with_origin:
        products_in_doc = doc_edges[doc_edges["source"] == doc_node]["target"].values
        doc_products[doc_node] = {p.replace("PRODUCT:", "", 1) for p in products_in_doc if p != origin_node}

    # Calcular Jaccard similarity entre pares de documentos
    from itertools import combinations
    similarities = []
    for doc1, doc2 in combinations(doc_products.keys(), 2):
        union = len(doc_products[doc1] | doc_products[doc2])
        intersection = len(doc_products[doc1] & doc_products[doc2])
        jaccard = intersection / union if union > 0 else 0
        similarities.append(jaccard)

    avg_jaccard = sum(similarities) / len(similarities) if similarities else 0
    volatility = round(1 - avg_jaccard, 4)  # Inverso: alta similitud → baja volatilidad

    # Clasificar
    if volatility > 0.7:
        volatility_class = "MUY VOLÁTIL (versátil, combina con muchos productos diferentes)"
    elif volatility > 0.4:
        volatility_class = "MODERADAMENTE VOLÁTIL (algunos patrones recurrentes)"
    else:
        volatility_class = "ESTABLE (aparece con conjunto consistente de productos)"

    # Documentos más comunes
    nodes = read_csv(dataset_id, f"transaction_graph_{graph_type}_nodes.csv")
    node_labels = {str(row.node_id): str(row.label) for row in nodes.itertuples(index=False)}
    origin_name = node_labels.get(origin_node, pid)

    sample_docs = sorted(doc_products.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    doc_samples = [
        {
            "document": doc,
            "co_products_count": len(prods),
            "co_products": ", ".join(sorted(prods)[:5])
        }
        for doc, prods in sample_docs
    ]

    return QueryResponse(
        answer=f"'{origin_name}' presenta volatilidad {volatility:.2%} ({volatility_class}). Aparece en {len(documents_with_origin)} documentos con Jaccard promedio {avg_jaccard:.2%}.",
        algorithm="Volatility Analysis - Jaccard similarity entre DOCUMENT product sets",
        table=doc_samples,
        metrics={
            "product_id": pid,
            "documents_count": int(len(documents_with_origin)),
            "avg_jaccard_similarity": round(avg_jaccard, 4),
            "volatility_score": volatility,
            "volatility_class": volatility_class,
        },
        evidence={
            "dataset_id": dataset_id,
            "artifacts": [f"transaction_graph_{graph_type}_edges.csv"],
            "graph": f"G_{graph_type}",
            "method": "Jaccard(doc1_products, doc2_products) over all pairs"
        },
    )


def document_logistics_efficiency(dataset_id: str, graph_type: str = "sales") -> QueryResponse:
    """
    PUNTO 3: Eficiencia logística por documento.

    Analiza patrones de documentos:
    - ¿Cuántos productos típicamente van en un documento?
    - ¿Cuál es la distribución (simples vs. complejos)?
    - ¿Cuánto volumen/monto promedio?
    """
    if graph_type not in ("sales", "purchases"):
        graph_type = "sales"

    try:
        edges = read_csv(dataset_id, f"transaction_graph_{graph_type}_edges.csv")
    except FileNotFoundError:
        return QueryResponse(ok=False, error=f"Grafo G_{graph_type} no disponible.", algorithm="Logistics Efficiency")

    edges["amount"] = pd.to_numeric(edges["amount"], errors="coerce").fillna(0)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0)

    # Filtrar solo aristas document -> product
    doc_edges = edges[edges["edge_type"].str.contains("line", na=False)].copy()

    if doc_edges.empty:
        return QueryResponse(ok=False, error="No hay documentos en este grafo.", algorithm="Logistics Efficiency")

    # Agrupar por documento
    doc_stats = doc_edges.groupby("source").agg(
        product_count=("target", "count"),
        total_quantity=("weight", "sum"),
        total_amount=("amount", "sum"),
    ).reset_index()
    doc_stats.columns = ["document", "product_count", "total_quantity", "total_amount"]

    # Distribución
    dist = doc_stats["product_count"].value_counts().sort_index()

    return QueryResponse(
        answer=f"Análisis de {len(doc_stats)} documentos en G_{graph_type}. Promedio {doc_stats['product_count'].mean():.1f} productos/documento.",
        algorithm="Logistics Efficiency - Document aggregation & distribution analysis",
        table=[
            {
                "metric": "Documentos analizados",
                "value": int(len(doc_stats))
            },
            {
                "metric": "Productos promedio por documento",
                "value": round(doc_stats["product_count"].mean(), 2)
            },
            {
                "metric": "Documentos simples (1-2 productos)",
                "value": int(doc_stats[doc_stats["product_count"] <= 2].shape[0])
            },
            {
                "metric": "Documentos complejos (10+ productos)",
                "value": int(doc_stats[doc_stats["product_count"] >= 10].shape[0])
            },
            {
                "metric": "Monto promedio por documento",
                "value": round(doc_stats["total_amount"].mean(), 2)
            },
            {
                "metric": "Monto máximo en un documento",
                "value": round(doc_stats["total_amount"].max(), 2)
            },
        ],
        metrics={
            "total_documents": int(len(doc_stats)),
            "avg_products_per_doc": round(doc_stats["product_count"].mean(), 2),
            "avg_amount_per_doc": round(doc_stats["total_amount"].mean(), 2),
            "distribution": {
                "1_product": int((doc_stats["product_count"] == 1).sum()),
                "2_3_products": int(((doc_stats["product_count"] >= 2) & (doc_stats["product_count"] <= 3)).sum()),
                "4_9_products": int(((doc_stats["product_count"] >= 4) & (doc_stats["product_count"] <= 9)).sum()),
                "10_plus_products": int((doc_stats["product_count"] >= 10).sum()),
            },
            "max_products_in_doc": int(doc_stats["product_count"].max()),
        },
        evidence={
            "dataset_id": dataset_id,
            "artifacts": [f"transaction_graph_{graph_type}_edges.csv"],
            "graph": f"G_{graph_type}",
        },
    )


def best_savings_by_document(dataset_id: str, limit: int = 15) -> QueryResponse:
    """
    PUNTO 4: Mejores ahorros considerando co-compras en documentos.

    Mejora Bellman-Ford actual: en lugar de ahorro por entidad-producto promedio,
    busca documentos donde múltiples productos se compraron juntos (mismo proveedor)
    y calcula ahorros considerando esa co-compra.
    """
    try:
        edges = read_csv(dataset_id, "transaction_graph_purchases_edges.csv")
        candidates = read_csv(dataset_id, "bellman_ford_candidates.csv")
    except FileNotFoundError:
        return QueryResponse(
            ok=False,
            error="Datos de Bellman-Ford no disponibles.",
            algorithm="Bellman-Ford Document Co-occurrence"
        )

    edges["amount"] = pd.to_numeric(edges["amount"], errors="coerce").fillna(0)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0)
    candidates["savings_per_unit"] = pd.to_numeric(candidates["savings_per_unit"], errors="coerce").fillna(0)

    # Encontrar documentos (nodos PURCHASE_DOC)
    doc_edges = edges[edges["edge_type"] == "purchase_line"].copy()

    if doc_edges.empty:
        return QueryResponse(
            ok=False,
            error="No hay documentos en purchase graph.",
            algorithm="Bellman-Ford Document Co-occurrence"
        )

    # Por cada documento, agrupar productos
    doc_products: dict[str, list[dict]] = {}
    for row in doc_edges.itertuples(index=False):
        doc = row.source
        prod = row.target.replace("PRODUCT:", "", 1)
        if doc not in doc_products:
            doc_products[doc] = []
        doc_products[doc].append({
            "product_id": prod,
            "quantity": float(row.weight or 0),
            "amount": float(row.amount or 0),
        })

    # Calcular ahorro potencial por documento
    doc_savings = []
    for doc, products in doc_products.items():
        if len(products) < 2:
            continue

        total_products = len(products)
        total_amount = sum(p["amount"] for p in products)
        savings_in_doc = 0

        for p in products:
            pid = p["product_id"]
            candidate = candidates[candidates["product_id"].astype(str) == pid]
            if not candidate.empty:
                savings_per_unit = float(candidate.iloc[0].get("savings_per_unit", 0))
                savings_in_doc += savings_per_unit * p["quantity"]

        if savings_in_doc > 0:
            doc_savings.append({
                "document": doc,
                "product_count": total_products,
                "total_amount": total_amount,
                "total_savings": savings_in_doc,
                "avg_savings_per_product": savings_in_doc / total_products,
                "savings_pct": round((savings_in_doc / total_amount * 100) if total_amount > 0 else 0, 2),
            })

    if not doc_savings:
        return QueryResponse(
            ok=False,
            error="No se encontraron documentos con potencial de ahorro.",
            algorithm="Bellman-Ford Document Co-occurrence"
        )

    ranked = sorted(doc_savings, key=lambda x: x["total_savings"], reverse=True)[:limit]

    table = [
        {
            "document": r["document"],
            "products": r["product_count"],
            "total_amount": round(r["total_amount"], 2),
            "potential_savings": round(r["total_savings"], 2),
            "savings_pct": f"{r['savings_pct']:.1f}%",
        }
        for r in ranked
    ]

    return QueryResponse(
        answer=f"Análisis de {len(doc_products)} documentos de compra. Top {len(ranked)} documentos con mayor potencial de ahorro.",
        algorithm="Bellman-Ford mejorado - Agregación por DOCUMENT nodes",
        table=table,
        metrics={
            "total_documents_analyzed": int(len(doc_products)),
            "documents_with_savings_potential": len(doc_savings),
            "top_savings_document": round(ranked[0]["total_savings"], 2) if ranked else 0,
            "total_potential_savings": round(sum(r["total_savings"] for r in doc_savings), 2),
        },
        evidence={
            "dataset_id": dataset_id,
            "artifacts": ["transaction_graph_purchases_edges.csv", "bellman_ford_candidates.csv"],
            "graph": "G_purchases (document aggregation)",
            "method": "Bellman-Ford candidates + DOCUMENT node grouping"
        },
    )


def document_concentration_analysis(dataset_id: str, graph_type: str = "sales") -> QueryResponse:
    """
    PUNTO 5: Análisis de concentración de líneas.

    ¿El negocio crece por volumen (muchos documentos simples) o por diversidad (documentos complejos)?
    - Métrica: Coeficiente Gini de productos por documento
    """
    if graph_type not in ("sales", "purchases"):
        graph_type = "sales"

    try:
        edges = read_csv(dataset_id, f"transaction_graph_{graph_type}_edges.csv")
    except FileNotFoundError:
        return QueryResponse(ok=False, error=f"Grafo G_{graph_type} no disponible.", algorithm="Concentration Analysis")

    doc_edges = edges[edges["edge_type"].str.contains("line", na=False)]

    if doc_edges.empty:
        return QueryResponse(ok=False, error="No hay documentos.", algorithm="Concentration Analysis")

    # Contar productos por documento
    doc_product_counts = doc_edges.groupby("source")["target"].count().values

    # Calcular Gini coefficient
    sorted_counts = sorted(doc_product_counts)
    n = len(sorted_counts)
    gini = (2 * sum((i + 1) * val for i, val in enumerate(sorted_counts))) / (n * sum(sorted_counts)) - (n + 1) / n
    gini = max(0, min(1, gini))  # Normalizar a [0, 1]

    # Clasificación
    if gini > 0.6:
        concentration = "MUY CONCENTRADA (pocos documentos con muchos productos dominan el volumen)"
    elif gini > 0.3:
        concentration = "MODERADAMENTE CONCENTRADA (mezcla de documentos simples y complejos)"
    else:
        concentration = "DISTRIBUIDA (documentos similares en tamaño)"

    # Estadísticas
    percentiles = {
        "p10": int(sorted(doc_product_counts)[int(n * 0.1)] if n > 10 else sorted_counts[0]),
        "p50": int(sorted_counts[int(n * 0.5)]),
        "p90": int(sorted_counts[int(n * 0.9)] if n > 10 else sorted_counts[-1]),
    }

    return QueryResponse(
        answer=f"Distribución de {len(doc_product_counts)} documentos. Coeficiente Gini: {gini:.4f} ({concentration})",
        algorithm="Concentration Analysis - Gini coefficient on DOCUMENT product distribution",
        table=[
            {
                "metric": "Total documentos",
                "value": int(n)
            },
            {
                "metric": "Productos por documento (media)",
                "value": round(sum(doc_product_counts) / n, 2)
            },
            {
                "metric": "Productos por documento (mediana)",
                "value": percentiles["p50"]
            },
            {
                "metric": "Productos por documento (min)",
                "value": int(min(doc_product_counts))
            },
            {
                "metric": "Productos por documento (max)",
                "value": int(max(doc_product_counts))
            },
            {
                "metric": "Coeficiente Gini",
                "value": round(gini, 4)
            },
            {
                "metric": "Tipo distribución",
                "value": concentration
            },
        ],
        metrics={
            "total_documents": int(n),
            "gini_coefficient": round(gini, 4),
            "mean_products_per_doc": round(sum(doc_product_counts) / n, 2),
            "median_products_per_doc": percentiles["p50"],
            "percentiles": percentiles,
        },
        evidence={
            "dataset_id": dataset_id,
            "artifacts": [f"transaction_graph_{graph_type}_edges.csv"],
            "graph": f"G_{graph_type}",
            "method": "Gini coefficient = (2 * Σ(i+1)*x_i) / (n*Σx_i) - (n+1)/n"
        },
    )

