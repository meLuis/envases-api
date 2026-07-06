from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import pandas as pd

from app.config import MIN_UFDS_SIMILARITY


class UFDS:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def families_from_projection(edges: pd.DataFrame, min_similarity: float = MIN_UFDS_SIMILARITY) -> pd.DataFrame:
    uf = UFDS()
    names: dict[str, str] = {}
    if edges.empty:
        return pd.DataFrame(columns=["family_id", "product_id", "product_name", "family_size"])
    for row in edges.itertuples(index=False):
        if float(row.similarity) < min_similarity:
            continue
        left = str(row.source)
        right = str(row.target)
        uf.union(left, right)
        names[left] = str(row.source_name)
        names[right] = str(row.target_name)
    groups: dict[str, list[str]] = defaultdict(list)
    for product_id in names:
        groups[uf.find(product_id)].append(product_id)
    rows = []
    for index, members in enumerate(sorted(groups.values(), key=len, reverse=True), start=1):
        for product_id in sorted(members):
            rows.append(
                {
                    "family_id": f"F{index:03d}",
                    "product_id": product_id,
                    "product_name": names.get(product_id, ""),
                    "family_size": len(members),
                }
            )
    return pd.DataFrame(rows)


def knapsack_budget(products: pd.DataFrame, items: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    """Knapsack sobre catálogo de productos cuando no hay supply_options.

    Usa unit_cost_purchase (precio de compra con IGV) del catálogo como costo real.
    Si el catálogo no tiene precio, usa 1.0 por unidad (budget = conteo de unidades).
    """
    product_lookup = products.set_index("product_id").to_dict("index") if not products.empty else {}
    expanded = []
    has_real_costs = False
    for item in items:
        product_id = str(item["product_id"])
        quantity = int(max(float(item.get("quantity", 1)), 1))
        row = product_lookup.get(product_id, {})
        value = float(item.get("value") or quantity)
        stock = int(max(float(row.get("stock") or quantity), 1))
        max_units = min(quantity, stock)
        unit_value = value / max(quantity, 1)
        # Precio de compra con IGV del catálogo; si no existe, costo unitario = 1
        catalog_cost = float(row.get("unit_cost_purchase") or 0)
        unit_cost = catalog_cost if catalog_cost > 0 else 1.0
        if catalog_cost > 0:
            has_real_costs = True
        for unit in range(max_units):
            expanded.append({"product_id": product_id, "unit": unit + 1, "value": unit_value, "cost": unit_cost})

    # Escalar presupuesto a centavos para trabajar con enteros
    scale = 100 if has_real_costs else 1
    capacity = int(max(budget, 0) * scale)
    dp = [0.0] * (capacity + 1)
    keep = [[False] * (capacity + 1) for _ in expanded]
    for i, item in enumerate(expanded):
        cost = min(int(round(item["cost"] * scale)), capacity + 1)
        for w in range(capacity, cost - 1, -1):
            candidate = dp[w - cost] + float(item["value"])
            if candidate > dp[w]:
                dp[w] = candidate
                keep[i][w] = True
    chosen = []
    w = capacity
    for i in range(len(expanded) - 1, -1, -1):
        if keep[i][w]:
            chosen.append(expanded[i])
            w -= min(int(round(expanded[i]["cost"] * scale)), capacity + 1)
    counts: dict[str, int] = defaultdict(int)
    total_cost = 0.0
    for item in chosen:
        counts[item["product_id"]] += 1
        total_cost += item["cost"]
    plan = [{"product_id": pid, "units": units} for pid, units in sorted(counts.items())]
    # Ítems candidatos (costo + valor por unidad) para reconstruir la tabla DP en el front.
    items_view = [
        {"product_id": it["product_id"], "cost": round(float(it["cost"]), 2), "value": round(float(it["value"]), 4)}
        for it in expanded[:12]
    ]
    return {
        "method": "knapsack_dp",
        "plan": plan,
        "items": items_view,
        "total_units": sum(counts.values()),
        "total_cost": round(total_cost, 2),
        "budget_left": round(budget - total_cost, 2),
        "score": round(dp[capacity], 4),
        "cost_source": "catalog_unit_cost_purchase" if has_real_costs else "unit_count_fallback",
    }


def knapsack_supply_budget(options: pd.DataFrame, items: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    """0/1 knapsack sobre lotes proveedor-producto con costos reales.

    Cada item DP representa un lote disponible para un producto solicitado. El
    valor por defecto es unidades cubiertas; si el request trae value, se reparte
    proporcionalmente por unidad solicitada.
    """
    requested = {str(item["product_id"]): float(item.get("quantity", 1)) for item in items}
    values = {
        str(item["product_id"]): float(item.get("value") or item.get("quantity", 1))
        for item in items
    }
    lots = []
    for product_id, requested_units in requested.items():
        remaining_units = int(max(requested_units, 0))
        product_options = options.loc[options["product_id"].astype(str) == product_id].sort_values("unit_cost")
        unit_value = values[product_id] / max(requested_units, 1.0)
        for row in product_options.itertuples(index=False):
            if remaining_units <= 0:
                break
            available = int(max(float(row.capacity_units), 0))
            if available <= 0 or float(row.unit_cost) <= 0:
                continue
            assignable = min(available, remaining_units)
            for _ in range(assignable):
                lots.append(
                    {
                        "product_id": product_id,
                        "supplier": str(row.supplier),
                        "units": 1.0,
                        "unit_cost": float(row.unit_cost),
                        "cost": float(row.unit_cost),
                        "value": unit_value,
                    }
                )
            remaining_units -= assignable
    capacity = int(round(max(budget, 0) * 100))
    if not lots or capacity <= 0:
        return {"method": "knapsack_dp", "plan": [], "items": [], "total_units": 0, "total_cost": 0.0, "budget_left": budget, "score": 0.0}
    dp = [0.0] * (capacity + 1)
    keep = [[False] * (capacity + 1) for _ in lots]
    costs = [min(int(round(lot["cost"] * 100)), capacity + 1) for lot in lots]
    for i, lot in enumerate(lots):
        cost = costs[i]
        for w in range(capacity, cost - 1, -1):
            candidate = dp[w - cost] + float(lot["value"])
            if candidate > dp[w]:
                dp[w] = candidate
                keep[i][w] = True
    chosen = []
    w = capacity
    for i in range(len(lots) - 1, -1, -1):
        if keep[i][w]:
            chosen.append(lots[i])
            w -= costs[i]
    chosen.reverse()
    aggregate: dict[tuple[str, str, float], dict[str, Any]] = {}
    for item in chosen:
        key = (item["product_id"], item["supplier"], item["unit_cost"])
        row = aggregate.setdefault(
            key,
            {
                "product_id": item["product_id"],
                "supplier": item["supplier"],
                "units": 0.0,
                "unit_cost": item["unit_cost"],
                "cost": 0.0,
            },
        )
        row["units"] += item["units"]
        row["cost"] += item["cost"]
    total_cost = sum(item["cost"] for item in chosen)
    items_view = [
        {"product_id": lot["product_id"], "cost": round(float(lot["cost"]), 2), "value": round(float(lot["value"]), 4)}
        for lot in lots[:12]
    ]
    return {
        "method": "knapsack_dp",
        "plan": [
            {key: round(value, 2) if isinstance(value, float) else value for key, value in item.items() if key != "value"}
            for item in aggregate.values()
        ],
        "items": items_view,
        "total_units": round(sum(item["units"] for item in chosen), 2),
        "total_cost": round(total_cost, 2),
        "budget_left": round(budget - total_cost, 2),
        "score": round(dp[capacity], 4),
    }


def min_cost_flow_supply(options: pd.DataFrame, demand: list[dict[str, Any]]) -> dict[str, Any]:
    """Asignación de demanda a proveedores con costo y capacidad (Min-cost flow).

    Red: SOURCE → producto (cap = demanda) → (producto, proveedor) (cap =
    capacidad del proveedor para ese producto, costo = costo unitario) →
    proveedor (cap = capacidad total del proveedor) → SINK. Resuelve con
    caminos de costo mínimo sucesivos (SPFA/Bellman-Ford), garantizando el
    costo total mínimo para el flujo servido. Respeta demanda y capacidad.
    """
    requested = {str(item["product_id"]): int(max(float(item.get("quantity", 0)), 0)) for item in demand}
    requested = {pid: qty for pid, qty in requested.items() if qty > 0}
    if options.empty or not requested:
        return {"ok": False, "reason": "empty", "assignment": [], "total_cost": 0.0, "served_units": 0, "demand_units": sum(requested.values()), "unmet": []}

    # Índice de nodos del grafo de flujo.
    node_id: dict[str, int] = {}

    def nid(name: str) -> int:
        if name not in node_id:
            node_id[name] = len(node_id)
        return node_id[name]

    SOURCE = nid("SOURCE")
    SINK = nid("SINK")

    # Aristas dirigidas como [to, capacity, cost, flow]; cada arista i tiene su
    # inversa en i^1 (residual) para poder devolver flujo.
    adj: dict[int, list[int]] = defaultdict(list)
    edges: list[list[float]] = []

    def add_edge(u: int, v: int, cap: float, cost: float) -> None:
        adj[u].append(len(edges))
        edges.append([v, cap, cost, 0.0])
        adj[v].append(len(edges))
        edges.append([u, 0.0, -cost, 0.0])

    # SOURCE → producto (capacidad = demanda).
    for pid, qty in requested.items():
        add_edge(SOURCE, nid(f"P:{pid}"), float(qty), 0.0)

    # producto → (producto, proveedor) con costo unitario; y proveedor → SINK
    # con la capacidad total del proveedor.
    supplier_cap: dict[str, float] = {}
    for row in options.itertuples(index=False):
        pid = str(getattr(row, "product_id", ""))
        if pid not in requested:
            continue
        supplier = str(getattr(row, "supplier", "") or getattr(row, "supplier_id", ""))
        cost = float(getattr(row, "unit_cost", 0) or 0)
        cap = float(getattr(row, "capacity_units", 0) or 0)
        if not supplier or cost <= 0 or cap <= 0:
            continue
        ps_node = nid(f"PS:{pid}:{supplier}")
        add_edge(nid(f"P:{pid}"), ps_node, cap, cost)
        add_edge(ps_node, nid(f"S:{supplier}"), cap, 0.0)
        supplier_cap[supplier] = max(supplier_cap.get(supplier, 0.0), float(getattr(row, "supplier_capacity", 0) or 0) or cap)

    for supplier, cap in supplier_cap.items():
        add_edge(nid(f"S:{supplier}"), SINK, cap, 0.0)

    total_nodes = len(node_id)

    # Min-cost max-flow con SPFA (Bellman-Ford por cola) por camino aumentante.
    def spfa() -> tuple[list[float], list[int]]:
        dist = [float("inf")] * total_nodes
        in_queue = [False] * total_nodes
        prev_edge = [-1] * total_nodes
        dist[SOURCE] = 0.0
        queue = deque([SOURCE])
        in_queue[SOURCE] = True
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for eidx in adj[u]:
                v, cap, cost, flow = edges[eidx]
                if cap - flow > 1e-9 and dist[u] + cost < dist[int(v)] - 1e-12:
                    dist[int(v)] = dist[u] + cost
                    prev_edge[int(v)] = eidx
                    if not in_queue[int(v)]:
                        queue.append(int(v))
                        in_queue[int(v)] = True
        return dist, prev_edge

    total_cost = 0.0
    served = 0.0
    while True:
        dist, prev_edge = spfa()
        if dist[SINK] == float("inf"):
            break
        # Cuello de botella del camino.
        bottleneck = float("inf")
        v = SINK
        while v != SOURCE:
            eidx = prev_edge[v]
            cap, flow = edges[eidx][1], edges[eidx][3]
            bottleneck = min(bottleneck, cap - flow)
            v = edges[eidx ^ 1][0]
            v = int(v)
        v = SINK
        while v != SOURCE:
            eidx = prev_edge[v]
            edges[eidx][3] += bottleneck
            edges[eidx ^ 1][3] -= bottleneck
            v = int(edges[eidx ^ 1][0])
        served += bottleneck
        total_cost += bottleneck * dist[SINK]

    # Reconstruir asignación: aristas dirigidas producto→PS con flujo positivo.
    assignment = []
    rev_index = {v: k for k, v in node_id.items()}
    for eidx in range(0, len(edges), 2):
        v, cap, cost, flow = edges[eidx]
        if flow <= 1e-9:
            continue
        u = int(edges[eidx ^ 1][0])
        u_name = rev_index.get(u, "")
        v_name = rev_index.get(int(v), "")
        if u_name.startswith("P:") and v_name.startswith("PS:"):
            _, pid, supplier = v_name.split(":", 2)
            assignment.append(
                {
                    "product_id": pid,
                    "supplier": supplier,
                    "units": round(flow, 2),
                    "unit_cost": round(cost, 4),
                    "line_cost": round(flow * cost, 2),
                }
            )
    assignment.sort(key=lambda item: (item["product_id"], item["line_cost"]))

    served_by_product: dict[str, float] = defaultdict(float)
    for item in assignment:
        served_by_product[item["product_id"]] += item["units"]
    unmet = [
        {"product_id": pid, "requested": qty, "served": int(served_by_product.get(pid, 0)), "shortfall": int(qty - served_by_product.get(pid, 0))}
        for pid, qty in requested.items()
        if served_by_product.get(pid, 0) < qty
    ]

    return {
        "ok": bool(assignment),
        "reason": "ok" if assignment else "infeasible",
        "assignment": assignment,
        "total_cost": round(total_cost, 2),
        "served_units": int(round(served)),
        "demand_units": sum(requested.values()),
        "unmet": unmet,
        "suppliers_used": len({item["supplier"] for item in assignment}),
    }


def build_supply_options(purchases: pd.DataFrame) -> pd.DataFrame:
    if purchases.empty:
        return pd.DataFrame()
    grouped = (
        purchases.loc[purchases["quantity"] > 0]
        .assign(unit_cost=lambda df: df["unit_value"].where(df["unit_value"] > 0, df["total"] / df["quantity"]))
        .groupby(["product_id", "product_name", "entity_norm", "entity_name"], as_index=False)
        .agg(
            unit_cost=("unit_cost", "median"),
            capacity_units=("quantity", "sum"),
            purchase_lines=("product_id", "size"),
            last_purchase=("date", "max"),
        )
    )
    supplier_caps = grouped.groupby("entity_norm")["capacity_units"].sum().to_dict()
    grouped["supplier_capacity"] = grouped["entity_norm"].map(supplier_caps)
    grouped = grouped.rename(columns={"entity_norm": "supplier_id", "entity_name": "supplier"})
    return grouped.sort_values(["product_id", "unit_cost"])


def bellman_ford_savings(options: pd.DataFrame) -> dict[str, Any]:
    if options.empty:
        return {"candidates": pd.DataFrame(), "edges": pd.DataFrame(), "best_paths": pd.DataFrame(), "summary": {"status": "empty"}}

    refs = options.groupby("product_id", as_index=False).agg(reference_unit_cost=("unit_cost", "median"))
    candidates = options.merge(refs, on="product_id", how="left")
    candidates["edge_weight"] = candidates["unit_cost"] - candidates["reference_unit_cost"]
    candidates["savings_per_unit"] = candidates["reference_unit_cost"] - candidates["unit_cost"]
    candidates["savings_pct"] = (
        candidates["savings_per_unit"] / candidates["reference_unit_cost"].replace(0, pd.NA)
    )

    # Construir grafo dirigido: SOURCE → PRODUCT_i (w=0) → OPTION_ij (w=edge_weight)
    edge_list: list[tuple[str, str, float, str, str]] = []  # (u, v, w, product_id, supplier)
    for row in candidates.itertuples(index=False):
        product_node = f"PRODUCT:{row.product_id}"
        option_node = f"OPTION:{row.product_id}:{row.supplier_id}"
        edge_list.append(("SOURCE", product_node, 0.0, str(row.product_id), ""))
        edge_list.append((product_node, option_node, float(row.edge_weight), str(row.product_id), str(row.supplier)))

    # Recopilar todos los nodos
    nodes: set[str] = {"SOURCE"}
    for u, v, *_ in edge_list:
        nodes.add(u)
        nodes.add(v)

    # Inicializar distancias Bellman-Ford
    dist: dict[str, float] = {node: float("inf") for node in nodes}
    dist["SOURCE"] = 0.0
    predecessor: dict[str, tuple[str, float, str, str] | None] = {node: None for node in nodes}

    # Relajación: |V| - 1 iteraciones
    num_vertices = len(nodes)
    for _ in range(num_vertices - 1):
        updated = False
        for u, v, w, product_id, supplier in edge_list:
            if dist[u] < float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                predecessor[v] = (u, w, product_id, supplier)
                updated = True
        if not updated:
            break  # convergencia anticipada

    # Construir DataFrame de aristas para trazabilidad
    edges_df = pd.DataFrame(
        [
            {"source": u, "target": v, "weight": w, "edge_type": "start_product" if u == "SOURCE" else "supplier_option", "product_id": pid, "supplier": sup}
            for u, v, w, pid, sup in edge_list
        ]
    )

    # Extraer el mejor camino por producto (opción con menor dist[OPTION_node])
    best_rows = []
    seen_products: set[str] = set()
    for row in candidates.sort_values("edge_weight").itertuples(index=False):
        product_id = str(row.product_id)
        if product_id in seen_products:
            continue
        option_node = f"OPTION:{row.product_id}:{row.supplier_id}"
        final_dist = dist.get(option_node, float("inf"))
        if final_dist < float("inf"):
            best_rows.append({
                "product_id": row.product_id,
                "product_name": row.product_name,
                "supplier": row.supplier,
                "unit_cost": round(float(row.unit_cost), 4),
                "reference_unit_cost": round(float(row.reference_unit_cost), 4),
                "edge_weight": round(float(row.edge_weight), 4),
                "savings_per_unit": round(float(row.savings_per_unit), 4),
                "savings_pct": round(float(row.savings_pct), 4) if pd.notna(row.savings_pct) else None,
                "bellman_ford_dist": round(final_dist, 4),
            })
            seen_products.add(product_id)

    best_paths = pd.DataFrame(best_rows).sort_values("savings_per_unit", ascending=False) if best_rows else pd.DataFrame()
    negative_edges = int((candidates["edge_weight"] < 0).sum())

    return {
        "candidates": candidates,
        "edges": edges_df,
        "best_paths": best_paths,
        "summary": {
            "status": "ok",
            "algorithm": "Bellman-Ford",
            "vertices": num_vertices,
            "edges": len(edge_list),
            "iterations": num_vertices - 1,
            "negative_edges": negative_edges,
        },
    }
