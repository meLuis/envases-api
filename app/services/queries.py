from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from app.core.algorithms import UFDS, knapsack_budget, knapsack_supply_budget, min_cost_flow_supply
from app.core.geo import haversine_km
from app.core.graphs import a_star_route, bidirectional_bfs_path, dijkstra_max_weight_path
from app.core.semantic_search import SemanticSearchIndex
from app.core.text import normalize_text, similarity
from app.domain.schemas import QueryResponse
from app.storage.repository import read_csv, read_json


def dataset_summary(dataset_id: str) -> dict:
    return read_json(dataset_id, "dataset_summary.json")


def _product_meta(products: pd.DataFrame) -> dict[str, dict[str, str]]:
    """product_id → {name, stock, name_norm} desde products_clean."""
    meta: dict[str, dict[str, str]] = {}
    for row in products.itertuples(index=False):
        pid = str(getattr(row, "product_id", ""))
        meta[pid] = {
            "name": str(getattr(row, "product_name", "")),
            "stock": str(getattr(row, "stock", "")),
            "name_norm": str(getattr(row, "product_name_norm", "")),
        }
    return meta


def search_products(dataset_id: str, query: str, limit: int = 10) -> QueryResponse:
    products = read_csv(dataset_id, "products_clean.csv")
    meta = _product_meta(products)

    # Buscador semántico por capas sobre G_attr (BFS multi-semilla + coverage
    # boost + filtro numérico exacto). Si no hay grafo o el query no resuelve
    # ninguna semilla, se cae a similitud textual sobre el catálogo.
    graph_results: list[dict] = []
    seed_info: dict = {}
    graph_understood_query = False
    used_graph = False
    try:
        attr_nodes = read_csv(dataset_id, "semantic_attribute_graph_nodes.csv")
        attr_edges = read_csv(dataset_id, "semantic_attribute_graph_edges.csv")
        index = SemanticSearchIndex.from_frames(attr_nodes, attr_edges)
        graph_results = index.search(query, k=limit)
        seed_info = index.last_stats
        graph_understood_query = bool(seed_info.get("seeds") or seed_info.get("unresolved_exact_filters"))
        used_graph = bool(graph_results)
    except FileNotFoundError:
        pass

    if used_graph or graph_understood_query:
        rows = []
        for item in graph_results:
            pid = str(item["product"]).replace("PRODUCT:", "", 1)
            info = meta.get(pid, {})
            rows.append(
                {
                    "product_id": pid,
                    "product_name": info.get("name") or item.get("label", ""),
                    "score": item["relevance"],
                    "units_sold": item.get("units_sold", 0),
                    "seed_coverage": f"{item['seed_coverage']}/{item['total_seeds']}",
                    "stock": info.get("stock", ""),
                    "match_source": "graph",
                }
            )
        return QueryResponse(
            answer=f"Se encontraron {len(rows)} productos compatibles con '{query}'.",
            algorithm=(
                "Búsqueda semántica sobre G_attr: BFS multi-semilla con decaimiento por "
                "distancia, intersección estricta de conceptos resueltos, boost por cobertura "
                "y filtros numéricos exactos (capacidad/boca)."
            ),
            table=rows,
            metrics={
                "matches": len(rows),
                "limit": limit,
                "seeds": len(seed_info.get("seeds", [])),
                "seed_groups": seed_info.get("seed_groups", 0),
                "exact_filters": seed_info.get("exact_filters", 0),
                "expanded_nodes": seed_info.get("expanded_nodes", 0),
                "scored_products": seed_info.get("scored_products", 0),
                "strict_candidates": seed_info.get("strict_candidates", len(rows)),
                "unresolved_exact_filters": seed_info.get("unresolved_exact_filters", []),
            },
            evidence={
                "dataset_id": dataset_id,
                "artifacts": [
                    "products_clean.csv",
                    "semantic_attribute_graph_nodes.csv",
                    "semantic_attribute_graph_edges.csv",
                ],
                "graph": "G_attr",
            },
        )

    # Fallback textual (sin grafo o sin semillas resueltas).
    q = normalize_text(query)
    q_terms = set(q.split()) if q else set()
    rows = []
    for row in products.itertuples(index=False):
        name_norm = str(getattr(row, "product_name_norm", "") or "")
        text_score = similarity(q, name_norm)
        if q and q in name_norm:
            text_score = max(text_score, 0.98)
        if text_score < 0.35:
            continue
        if len(q_terms) > 1 and not all(term in set(name_norm.split()) for term in q_terms):
            continue
        rows.append(
            {
                "product_id": getattr(row, "product_id", ""),
                "product_name": getattr(row, "product_name", ""),
                "score": round(text_score, 4),
                "stock": getattr(row, "stock", ""),
                "match_source": "text",
            }
        )
    rows = sorted(rows, key=lambda item: item["score"], reverse=True)[:limit]
    return QueryResponse(
        answer=f"Se encontraron {len(rows)} productos compatibles con '{query}'.",
        algorithm="Similitud textual + filtro AND multi-término (sin grafo de atributos disponible).",
        table=rows,
        metrics={"matches": len(rows), "limit": limit, "seeds": 0},
        evidence={"dataset_id": dataset_id, "artifacts": ["products_clean.csv"], "graph": "G_attr"},
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


def logistics_a_star(dataset_id: str, client: str, supplier: str) -> QueryResponse:
    """A* logístico sobre la capa geográfica de G_business: ruta de menor
    distancia cliente → proveedor usando la heurística de distancia recta."""
    try:
        nodes = read_csv(dataset_id, "logistics_nodes.csv")
        edges = read_csv(dataset_id, "logistics_edges.csv")
        logistics_metrics = read_json(dataset_id, "logistics_metrics.json")
    except FileNotFoundError:
        return QueryResponse(
            ok=False,
            error="A* logístico requiere un dataset sintético/logístico con coordenadas (lat/lon). El dataset actual no las tiene.",
            algorithm="A* (heurística por distancia)",
            metrics={"logistics_available": False},
        )
    if edges.empty or nodes.empty:
        return QueryResponse(
            ok=False,
            error="La capa logística está vacía: se requieren coordenadas de clientes y proveedores.",
            algorithm="A* (heurística por distancia)",
            metrics={"logistics_available": False},
        )
    edges["km"] = pd.to_numeric(edges["km"], errors="coerce").fillna(0)
    coords: dict[str, tuple[float, float]] = {}
    labels: dict[str, str] = {}
    for row in nodes.itertuples(index=False):
        node_id = str(row.node_id)
        try:
            coords[node_id] = (float(row.lat), float(row.lon))
        except (TypeError, ValueError):
            continue
        labels[node_id] = str(getattr(row, "label", node_id) or node_id)

    start = _find_node(nodes, "CLIENT", client)
    goal = _find_node(nodes, "SUPPLIER", supplier)
    if not start or not goal:
        return QueryResponse(
            ok=False,
            error="Cliente o proveedor sin coordenadas en la capa logística.",
            algorithm="A* (heurística por distancia)",
            metrics={"logistics_available": True},
        )

    result = a_star_route(edges, coords, start, goal)
    if not result["ok"]:
        return QueryResponse(
            ok=False,
            error="No existe ruta logistica entre ese cliente y ese proveedor en la red vial sintetica.",
            algorithm="A* (heurística por distancia)",
            metrics={"logistics_available": True},
        )

    path = result["path"]
    g_by_node = {step["node"]: step["g_km"] for step in result["visit_order"]}
    lat_goal, lon_goal = coords[goal]

    def enrich_trace(items: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
        out = []
        selected = items if limit is None else items[:limit]
        for item in selected:
            node = str(item.get("node", ""))
            lat, lon = coords.get(node, (0.0, 0.0))

            def enrich_node(candidate: dict[str, Any]) -> dict[str, Any]:
                candidate_node = str(candidate.get("node", ""))
                candidate_lat, candidate_lon = coords.get(candidate_node, (0.0, 0.0))
                return {
                    **candidate,
                    "label": labels.get(candidate_node, candidate_node),
                    "lat": round(candidate_lat, 6),
                    "lon": round(candidate_lon, 6),
                }

            out.append(
                {
                    **item,
                    "label": labels.get(node, node),
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "neighbors": [enrich_node(n) for n in item.get("neighbors", [])],
                    "frontier": [enrich_node(n) for n in item.get("frontier", [])],
                }
            )
        return out

    def map_nodes() -> list[dict[str, Any]]:
        route_types = {"HUB", "ROUTE"}
        important = set(path)
        for item in result["visit_order"][:80]:
            important.add(str(item.get("node", "")))
            important.update(str(n.get("node", "")) for n in item.get("frontier", []))
            important.update(str(n.get("node", "")) for n in item.get("neighbors", []))
        rows = []
        for row in nodes.itertuples(index=False):
            node_id = str(row.node_id)
            node_type = str(getattr(row, "node_type", "") or "")
            if node_type not in route_types and node_id not in important:
                continue
            rows.append(
                {
                    "node": node_id,
                    "label": labels.get(node_id, node_id),
                    "type": node_type,
                    "lat": round(float(getattr(row, "lat", 0) or 0), 6),
                    "lon": round(float(getattr(row, "lon", 0) or 0), 6),
                    "zone": str(getattr(row, "zone", "") or ""),
                }
            )
        return rows

    def map_edges() -> list[dict[str, Any]]:
        path_pairs = {tuple(sorted((path[i - 1], path[i]))) for i in range(1, len(path))}
        rows = []
        for row in edges.itertuples(index=False):
            source = str(row.source)
            target = str(row.target)
            edge_type = str(getattr(row, "edge_type", "") or "")
            include = edge_type != "access" or tuple(sorted((source, target))) in path_pairs
            if not include:
                continue
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "km": round(float(getattr(row, "km", 0) or 0), 4),
                    "type": edge_type,
                    "road": str(getattr(row, "road_name", "") or ""),
                }
            )
        return rows

    table = []
    for index, node in enumerate(path):
        lat, lon = coords[node]
        table.append(
            {
                "step": index + 1,
                "node": node,
                "label": labels.get(node, node),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "g_km": g_by_node.get(node, 0.0),
                "h_km": round(haversine_km(lat, lon, lat_goal, lon_goal), 3),
            }
        )
    return QueryResponse(
        answer=(
            f"Ruta logística de {result['total_km']} km entre '{labels.get(start, client)}' y "
            f"'{labels.get(goal, supplier)}' ({len(path)} paradas). A* expandió {result['expanded']} nodos "
            f"frente a {result['baseline_expanded']} de Dijkstra sin heurística."
        ),
        algorithm="A* con costo real g(n)=km acumulados y heurística admisible h(n)=distancia recta al destino",
        table=table,
        metrics={
            "total_km": result["total_km"],
            "stops": len(path),
            "expanded_a_star": result["expanded"],
            "expanded_dijkstra": result["baseline_expanded"],
            "question_answered": "Ruta de menor distancia entre cliente y proveedor en la red logistica.",
            "business_use": "Planificar el corredor de entrega/recojo y mostrar cuanta exploracion evita la heuristica de A*.",
            "logistics_available": True,
            "astar_trace": enrich_trace(result["visit_order"]),
            "dijkstra_trace": enrich_trace(result["baseline_visit_order"], limit=120),
            "map_nodes": map_nodes(),
            "map_edges": map_edges(),
            "obstacles": logistics_metrics.get("obstacles", []),
            "logistics_model": logistics_metrics.get("model", "geographic_overlay"),
        },
        evidence={"dataset_id": dataset_id, "artifacts": ["logistics_nodes.csv", "logistics_edges.csv"], "graph": "G_business (overlay logístico)"},
    )


def min_cost_supply(dataset_id: str, items: list[dict]) -> QueryResponse:
    """Min-cost flow: asigna la demanda por producto a proveedores minimizando
    el costo total y respetando capacidades."""
    try:
        options = read_csv(dataset_id, "supply_options.csv")
    except FileNotFoundError:
        return QueryResponse(ok=False, error="No hay opciones de suministro (supply_options.csv).", algorithm="Min-cost flow")
    for col in ["unit_cost", "capacity_units", "supplier_capacity"]:
        if col in options.columns:
            options[col] = pd.to_numeric(options[col], errors="coerce").fillna(0)
    names = {str(r.product_id): str(getattr(r, "product_name", "")) for r in options.itertuples(index=False)} if "product_name" in options.columns else {}

    def with_name(row: dict) -> dict:
        return {**row, "product_name": names.get(row.get("product_id", ""), "")}

    result = min_cost_flow_supply(options, items)
    table = [with_name(row) for row in result["assignment"]]
    steps = [with_name(row) for row in result.get("steps", [])]
    network = result.get("network", {})
    network = {
        "source_edges": [with_name(row) for row in network.get("source_edges", [])],
        "flow_edges": [with_name(row) for row in network.get("flow_edges", [])],
        "sink_edges": network.get("sink_edges", []),
    }
    metrics = {key: value for key, value in result.items() if key not in ("assignment", "steps", "network")}
    metrics["steps"] = steps
    metrics["network"] = network
    return QueryResponse(
        ok=result["ok"],
        answer=(
            f"Demanda de {result['demand_units']} unidades cubierta con {result['served_units']} "
            f"(costo mínimo total S/{result['total_cost']}, {result.get('suppliers_used', 0)} proveedores)."
            if result["ok"]
            else "No se pudo asignar la demanda: sin capacidad o costo válido en las opciones."
        ),
        algorithm="Min-cost flow (caminos de costo mínimo sucesivos / SPFA): SOURCE→producto→proveedor→SINK",
        table=table,
        metrics=metrics,
        evidence={"dataset_id": dataset_id, "artifacts": ["supply_options.csv"], "graph": "flow"},
    )


def supplier_substitutes(dataset_id: str, supplier_id: str) -> QueryResponse:
    families = read_csv(dataset_id, "ufds_supplier_families.csv")
    projection = read_csv(dataset_id, "supplier_projection_edges.csv")
    matched = _resolve_supplier_id(supplier_id, families, projection)
    if matched is None:
        suggestions = _supplier_suggestions(supplier_id, families, projection)
        return QueryResponse(
            ok=False,
            error=(
                f"No se encontro un proveedor parecido a '{supplier_id}'. "
                f"Prueba con: {', '.join(suggestions[:5])}."
            ),
            algorithm="UFDS / componentes conexos sobre proveedores",
            metrics={"suggestions": suggestions[:5]},
        )

    supplier_rows = families.loc[families["supplier_id"].astype(str) == matched["supplier_id"]]
    if supplier_rows.empty:
        return QueryResponse(ok=False, error="Proveedor no pertenece a una familia UFDS generada.", algorithm="UFDS / componentes conexos")
    family_id = supplier_rows.iloc[0]["family_id"]
    projection["similarity"] = pd.to_numeric(projection["similarity"], errors="coerce").fillna(0)
    for col in ["shared_products", "shared_attributes", "source_coverage", "target_coverage"]:
        if col in projection.columns:
            projection[col] = pd.to_numeric(projection[col], errors="coerce").fillna(0)
    supplier_key = str(matched["supplier_id"])
    left = projection.loc[projection["source"].astype(str) == supplier_key].rename(
        columns={"target": "supplier_id", "target_name": "supplier_name"}
    )
    if not left.empty:
        left["coverage_of_query_catalog"] = left.get("source_coverage", 0)
        left["coverage_of_candidate_catalog"] = left.get("target_coverage", 0)
    right = projection.loc[projection["target"].astype(str) == supplier_key].rename(
        columns={"source": "supplier_id", "source_name": "supplier_name"}
    )
    if not right.empty:
        right["coverage_of_query_catalog"] = right.get("target_coverage", 0)
        right["coverage_of_candidate_catalog"] = right.get("source_coverage", 0)
    shared_col = "shared_products" if "shared_products" in projection.columns else "shared_attributes"
    related = (
        pd.concat([left, right], ignore_index=True)[
            ["supplier_id", "supplier_name", shared_col, "coverage_of_query_catalog", "coverage_of_candidate_catalog", "similarity"]
        ]
        .rename(columns={shared_col: "shared_products"})
        .sort_values(["similarity", "shared_products"], ascending=False)
        .head(20)
    )
    return QueryResponse(
        answer=(
            f"Proveedor '{matched['supplier_name']}' ubicado en familia {family_id}; "
            "se devuelven respaldos directos con catalogo compartido."
        ),
        algorithm="UFDS sobre proveedores: Jaccard de catalogo comprado + ranking por respaldo directo",
        table=_records(related.head(20)),
        metrics={
            "input_supplier": supplier_id,
            "matched_supplier_id": matched["supplier_id"],
            "matched_supplier_name": matched["supplier_name"],
            "match_score": matched["score"],
            "family_id": family_id,
            "family_size": int(supplier_rows.iloc[0]["family_size"]),
        },
        evidence={"dataset_id": dataset_id, "artifacts": ["ufds_supplier_families.csv", "supplier_projection_edges.csv"], "graph": "G_supplier_projection"},
    )


def _resolve_supplier_id(query: str, families: pd.DataFrame, projection: pd.DataFrame) -> dict[str, Any] | None:
    candidates = _supplier_candidates(families, projection)
    q = normalize_text(query)
    if not q:
        return None

    scored = []
    for item in candidates:
        supplier_id = str(item["supplier_id"])
        supplier_name = str(item["supplier_name"])
        id_norm = normalize_text(supplier_id)
        name_norm = normalize_text(supplier_name)
        text_score = max(similarity(q, id_norm), similarity(q, name_norm))
        if q == id_norm or q == name_norm:
            score = 1.0
        elif q in id_norm or q in name_norm:
            score = 0.92
        else:
            score = text_score
        scored.append({**item, "score": round(float(score), 4)})

    best = max(scored, key=lambda item: item["score"], default=None)
    if best is None or best["score"] < 0.5:
        return None
    return best


def _supplier_suggestions(query: str, families: pd.DataFrame, projection: pd.DataFrame) -> list[str]:
    candidates = _supplier_candidates(families, projection)
    q = normalize_text(query)
    ranked = sorted(
        candidates,
        key=lambda item: max(
            similarity(q, normalize_text(item["supplier_id"])),
            similarity(q, normalize_text(item["supplier_name"])),
        ),
        reverse=True,
    )
    return [str(item["supplier_name"] or item["supplier_id"]) for item in ranked[:5]]


def _supplier_candidates(families: pd.DataFrame, projection: pd.DataFrame) -> list[dict[str, str]]:
    rows: dict[str, str] = {}
    if not families.empty:
        for row in families.itertuples(index=False):
            supplier_id = str(getattr(row, "supplier_id", "") or "")
            if supplier_id:
                rows[supplier_id] = str(getattr(row, "supplier_name", "") or supplier_id)
    if not projection.empty:
        for row in projection.itertuples(index=False):
            source = str(getattr(row, "source", "") or "")
            target = str(getattr(row, "target", "") or "")
            if source:
                rows.setdefault(source, str(getattr(row, "source_name", "") or source))
            if target:
                rows.setdefault(target, str(getattr(row, "target_name", "") or target))
    return [{"supplier_id": supplier_id, "supplier_name": name} for supplier_id, name in rows.items()]


def optimize_budget(dataset_id: str, budget: float, items: list[dict]) -> QueryResponse:
    try:
        options = read_csv(dataset_id, "supply_options.csv")
        for col in ["unit_cost", "capacity_units", "supplier_capacity"]:
            options[col] = pd.to_numeric(options[col], errors="coerce").fillna(0)
        products = read_csv(dataset_id, "products_clean.csv")
        result = knapsack_supply_budget(options, items, budget, products)
        artifacts = ["supply_options.csv", "products_clean.csv"]
    except FileNotFoundError:
        products = read_csv(dataset_id, "products_clean.csv")
        result = knapsack_budget(products, items, budget)
        artifacts = ["products_clean.csv"]
    return QueryResponse(
        answer=(
            f"Con presupuesto S/ {budget:,.2f}, la mejor combinacion usa S/ {result['total_cost']:,.2f} "
            f"y deja S/ {result['budget_left']:,.2f}. Ganancia esperada maxima: S/ {result['score']:,.2f}."
        ),
        algorithm=(
            "Knapsack 0/1 por programacion dinamica: cada oferta de proveedor es un lote indivisible. "
            "La celda dp[i][w] guarda la mejor ganancia esperada usando las primeras i ofertas con presupuesto w."
        ),
        table=result["plan"],
        metrics={
            **{key: value for key, value in result.items() if key != "plan"},
            "question_answered": "Que ofertas de proveedores conviene aceptar para maximizar ganancia esperada sin exceder el presupuesto.",
            "business_use": "Comparar combinaciones completas: no basta elegir lo mas barato ni la oferta con mayor precio de venta.",
        },
        evidence={"dataset_id": dataset_id, "artifacts": artifacts},
    )


def best_savings(dataset_id: str, limit: int = 20) -> QueryResponse:
    best = read_csv(dataset_id, "bellman_ford_best_paths.csv")
    edges = read_csv(dataset_id, "bellman_ford_edges.csv")
    summary = read_json(dataset_id, "bellman_ford_summary.json")
    table = _records(best.head(limit))
    trace_edges: list[dict[str, Any]] = []
    if table and "path" in table[0] and not edges.empty:
        path = [node for node in str(table[0].get("path", "")).split("|") if node]
        edge_lookup = {
            (str(row.source), str(row.target)): row
            for row in edges.itertuples(index=False)
        }
        for index in range(1, len(path)):
            row = edge_lookup.get((path[index - 1], path[index]))
            if row is None:
                continue
            trace_edges.append(
                {
                    "source": path[index - 1],
                    "target": path[index],
                    "weight": round(float(getattr(row, "weight", 0) or 0), 4),
                    "edge_type": str(getattr(row, "edge_type", "") or ""),
                    "label": str(getattr(row, "label", "") or ""),
                }
            )
    first = table[0] if table else {}
    decision = "esperar" if first.get("decision") == "esperar" else "comprar hoy"
    return QueryResponse(
        answer=(
            f"Bellman-Ford recomienda {decision}: mejor alternativa {first.get('campaign', '')} "
            f"para {first.get('product_name', 'producto')} con ahorro estimado S/ {first.get('savings', 0)}."
            if table
            else "No hay campañas sinteticas evaluables."
        ),
        algorithm=(
            "Bellman-Ford sobre grafo de campanas sinteticas: costos positivos representan comprar/esperar; "
            "aristas negativas representan descuentos de campana o cupon. El camino de menor costo indica si conviene comprar hoy o esperar."
        ),
        table=table,
        metrics={
            **summary,
            "trace_edges": trace_edges,
            "question_answered": "Conviene comprar hoy o esperar una campana cercana para reducir el costo?",
            "business_use": "Evaluar descuentos combinables y costo de espera con pesos negativos, algo que Dijkstra no maneja correctamente.",
        },
        evidence={"dataset_id": dataset_id, "artifacts": ["bellman_ford_best_paths.csv", "bellman_ford_edges.csv"], "graph": "G_offers"},
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
        supplier_proj = read_json(dataset_id, "supplier_projection_metrics.json")
        result["by_graph"]["G_supplier_projection"] = {
            "nodes": supplier_proj.get("supplier_count", 0),
            "edges": supplier_proj.get("edge_count", 0),
        }
    except FileNotFoundError:
        pass
    try:
        offers = read_json(dataset_id, "bellman_ford_summary.json")
        result["by_graph"]["G_offers"] = {
            "nodes": offers.get("vertices", 0),
            "edges": offers.get("edges", 0),
        }
    except FileNotFoundError:
        pass
    try:
        options = read_csv(dataset_id, "supply_options.csv")
        products = options["product_id"].astype(str).nunique() if "product_id" in options else 0
        suppliers = options["supplier"].astype(str).nunique() if "supplier" in options else 0
        result["by_graph"]["flow"] = {
            "nodes": int(products + suppliers + 2),
            "edges": int(len(options) + products + suppliers),
        }
    except FileNotFoundError:
        pass
    # Disponibilidad logística (A*) sin exponer un octavo grafo en la galería.
    try:
        logistics = read_json(dataset_id, "logistics_metrics.json")
        result["logistics_available"] = bool(logistics.get("available"))
        result["logistics"] = {"nodes": logistics.get("node_count", 0), "edges": logistics.get("edge_count", 0)}
    except FileNotFoundError:
        result["logistics_available"] = False
    result["node_type_breakdown"] = type_counts
    result["total_unique_nodes"] = sum(type_counts.values())
    return result


def weighted_connection(dataset_id: str, source: str, target: str, graph_type: str = "business") -> QueryResponse:
    gt = graph_type if graph_type in ("business", "sales", "purchases") else "business"
    nodes = read_csv(dataset_id, f"transaction_graph_{gt}_nodes.csv")
    edges = read_csv(dataset_id, f"transaction_graph_{gt}_edges.csv")
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
        algorithm=f"Dijkstra ponderado por volumen comercial (amount) en G_{gt}",
        table=[{"step": i + 1, "node": node, "label": _label(nodes, node)} for i, node in enumerate(path)],
        metrics={"path_length": max(len(path) - 1, 0), "total_commercial_volume": total_weight},
        evidence={"dataset_id": dataset_id, "artifacts": [f"transaction_graph_{gt}_edges.csv"], "graph": f"G_{gt}"},
    )


def mst_kruskal(dataset_id: str, graph_type: str = "business", limit: int = 60) -> QueryResponse:
    """Arbol de expansion minima (Kruskal): ordena aristas por peso y usa UFDS para
    ir uniendo componentes, descartando las que formarian ciclo. Reusa la misma
    estructura Union-Find de los sustitutos de proveedor: Kruskal ES union-find en accion.

    - transaction graphs (business/sales/purchases): peso = amount (monto). MST = red
      minima de transacciones que mantiene a todos conectados.
    - supplier_projection: peso = 1 - similarity (distancia). MST = columna vertebral
      que enlaza a todos los proveedores por sus parecidos mas fuertes.

    Devuelve en `table` las aristas CONSIDERADAS en orden de peso (hasta `limit`),
    marcando accepted=True (unio dos componentes) o accepted=False (ciclo), para animar
    el proceso en el front.
    """
    gt = graph_type
    labels: dict[str, str] = {}
    if gt == "supplier_projection":
        raw = read_csv(dataset_id, "supplier_projection_edges.csv")
        if raw.empty:
            return QueryResponse(ok=False, error="G_supplier_projection no disponible.", algorithm="Kruskal MST")
        raw["similarity"] = pd.to_numeric(raw["similarity"], errors="coerce").fillna(0)
        weighted: list[tuple[str, str, float, float]] = []
        for r in raw.itertuples(index=False):
            s, t = str(r.source), str(r.target)
            sim = float(r.similarity or 0)
            weighted.append((s, t, 1.0 - sim, sim))
            labels[s] = str(getattr(r, "source_name", s) or s)
            labels[t] = str(getattr(r, "target_name", t) or t)
        graph_label = "G_supplier_projection"
        artifacts = ["supplier_projection_edges.csv"]
        weight_desc = "1 - similitud (distancia)"
    else:
        gt = gt if gt in ("business", "sales", "purchases") else "business"
        raw = read_csv(dataset_id, f"transaction_graph_{gt}_edges.csv")
        nodes = read_csv(dataset_id, f"transaction_graph_{gt}_nodes.csv")
        if raw.empty:
            return QueryResponse(ok=False, error=f"G_{gt} no disponible.", algorithm="Kruskal MST")
        raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce").fillna(0)
        # Colapsar aristas paralelas al menor monto (grafo simple no dirigido).
        best: dict[tuple[str, str], float] = {}
        for r in raw.itertuples(index=False):
            s, t = str(r.source), str(r.target)
            key = (s, t) if s <= t else (t, s)
            w = float(r.amount or 0)
            if key not in best or w < best[key]:
                best[key] = w
        weighted = [(k[0], k[1], w, w) for k, w in best.items()]
        labels = {str(r.node_id): str(r.label) for r in nodes.itertuples(index=False)}
        graph_label = f"G_{gt}"
        artifacts = [f"transaction_graph_{gt}_edges.csv", f"transaction_graph_{gt}_nodes.csv"]
        weight_desc = "monto (amount)"

    weighted.sort(key=lambda e: e[2])
    uf = UFDS()
    considered_all: list[dict[str, Any]] = []
    tree_edges = 0
    rejected = 0
    total_weight = 0.0
    node_set: set[str] = {s for s, _, _, _ in weighted} | {t for _, t, _, _ in weighted}
    target_tree_edges = max(len(node_set) - 1, 0)
    mst_completed_at: int | None = None
    for considered_step, (s, t, w, orig) in enumerate(weighted, start=1):
        accepted = uf.find(s) != uf.find(t)
        if accepted:
            uf.union(s, t)
            tree_edges += 1
            total_weight += w
            if tree_edges == target_tree_edges and mst_completed_at is None:
                mst_completed_at = considered_step
        else:
            rejected += 1
        considered_all.append(
            {
                "considered_step": considered_step,
                "source_id": s,
                "target_id": t,
                "source": labels.get(s, s),
                "target": labels.get(t, t),
                "weight": round(w, 4),
                "raw_weight": round(orig, 4),
                "accepted": accepted,
            }
        )
    components = len({uf.find(n) for n in node_set}) if node_set else 0

    if gt == "supplier_projection":
        accepted_rows = [row for row in considered_all if row["accepted"]]
        prefix_end = mst_completed_at or len(considered_all)
        cycle_rows = [
            row
            for row in considered_all
            if int(row["considered_step"]) <= prefix_end and not row["accepted"]
        ]
        cycle_budget = min(len(cycle_rows), max(4, min(12, limit // 3))) if cycle_rows else 0
        accepted_budget = max(0, min(len(accepted_rows), limit - cycle_budget))
        picked_accepted = accepted_rows[:accepted_budget]

        visible_uf = UFDS()
        for row in picked_accepted:
            visible_uf.union(str(row["source_id"]), str(row["target_id"]))

        picked_cycles: list[dict[str, Any]] = []
        used_cycle_steps: set[int] = set()
        for row in cycle_rows:
            if len(picked_cycles) >= cycle_budget:
                break
            source_id = str(row["source_id"])
            target_id = str(row["target_id"])
            if visible_uf.find(source_id) == visible_uf.find(target_id):
                picked_cycles.append(row)
                used_cycle_steps.add(int(row["considered_step"]))

        if len(picked_cycles) < cycle_budget:
            for row in cycle_rows:
                step_no = int(row["considered_step"])
                if step_no in used_cycle_steps:
                    continue
                picked_cycles.append(row)
                used_cycle_steps.add(step_no)
                if len(picked_cycles) >= cycle_budget:
                    break

        considered = sorted(
            picked_accepted + picked_cycles,
            key=lambda row: int(row["considered_step"]),
        )[:limit]
    else:
        considered = considered_all[:limit]

    for idx, row in enumerate(considered, start=1):
        row["step"] = idx

    return QueryResponse(
        ok=tree_edges > 0,
        answer=(
            f"MST sobre {graph_label}: {tree_edges} aristas conectan {len(node_set)} nodos "
            f"(peso total {round(total_weight, 2)}); se descartaron {rejected} por ciclo."
        ),
        algorithm=f"Kruskal (ordenar aristas + Union-Find) en {graph_label}; peso = {weight_desc}",
        table=considered,
        metrics={
            "nodes": len(node_set),
            "candidate_edges": len(weighted),
            "tree_edges": tree_edges,
            "rejected_by_cycle": rejected,
            "total_weight": round(total_weight, 2),
            "components": components,
            "considered_shown": len(considered),
            "considered_to_complete_mst": mst_completed_at or len(considered_all),
            "weight": weight_desc,
        },
        evidence={"dataset_id": dataset_id, "artifacts": artifacts, "graph": graph_label},
    )


def graph_components(dataset_id: str, graph_type: str = "business", limit: int = 20) -> QueryResponse:
    """Componentes conexas con Union-Find (UFDS): agrupa nodos alcanzables entre si.
    Misma estructura que sustitutos de proveedor y que Kruskal, aqui aplicada al grafo
    comercial para responder '¿que actores forman una misma red?'.
    """
    gt = graph_type if graph_type in ("business", "sales", "purchases") else "business"
    edges = read_csv(dataset_id, f"transaction_graph_{gt}_edges.csv")
    nodes = read_csv(dataset_id, f"transaction_graph_{gt}_nodes.csv")
    if edges.empty:
        return QueryResponse(ok=False, error=f"G_{gt} no disponible.", algorithm="Union-Find / componentes conexas")
    labels = {str(r.node_id): str(r.label) for r in nodes.itertuples(index=False)}
    ntypes = {str(r.node_id): str(r.node_type) for r in nodes.itertuples(index=False)}
    uf = UFDS()
    node_set: set[str] = set()
    for r in edges.itertuples(index=False):
        s, t = str(r.source), str(r.target)
        uf.union(s, t)
        node_set.add(s)
        node_set.add(t)
    groups: dict[str, list[str]] = defaultdict(list)
    for n in node_set:
        groups[uf.find(n)].append(n)
    ordered = sorted(groups.values(), key=len, reverse=True)
    table = []
    for idx, members in enumerate(ordered[:limit], start=1):
        types = defaultdict(int)
        for m in members:
            types[ntypes.get(m, "OTHER")] += 1
        sample = ", ".join(labels.get(m, m) for m in members[:4])
        table.append(
            {
                "component": f"C{idx:03d}",
                "size": len(members),
                "composition": ", ".join(f"{k}:{v}" for k, v in sorted(types.items())),
                "sample": sample + ("…" if len(members) > 4 else ""),
            }
        )
    return QueryResponse(
        ok=bool(ordered),
        answer=(
            f"{len(ordered)} componentes conexas en G_{gt}; la mayor agrupa "
            f"{len(ordered[0]) if ordered else 0} nodos de {len(node_set)}."
        ),
        algorithm=f"Union-Find (UFDS) sobre G_{gt}: une nodos por cada arista y agrupa por raiz",
        table=table,
        metrics={
            "total_components": len(ordered),
            "nodes": len(node_set),
            "largest_component": len(ordered[0]) if ordered else 0,
            "shown": len(table),
        },
        evidence={
            "dataset_id": dataset_id,
            "artifacts": [f"transaction_graph_{gt}_edges.csv", f"transaction_graph_{gt}_nodes.csv"],
            "graph": f"G_{gt}",
        },
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
