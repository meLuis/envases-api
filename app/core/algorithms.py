from __future__ import annotations

from collections import defaultdict
import heapq
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


class MinCostFlow:
    def __init__(self) -> None:
        self.graph: list[list[list[float]]] = []
        self.ids: dict[str, int] = {}
        self.names: list[str] = []

    def node(self, name: str) -> int:
        if name not in self.ids:
            self.ids[name] = len(self.names)
            self.names.append(name)
            self.graph.append([])
        return self.ids[name]

    def add_edge(self, source: str, target: str, capacity: float, cost: float) -> None:
        u = self.node(source)
        v = self.node(target)
        self.graph[u].append([v, capacity, cost, len(self.graph[v])])
        self.graph[v].append([u, 0.0, -cost, len(self.graph[u]) - 1])

    def solve(self, source_name: str, sink_name: str) -> dict[str, Any]:
        source = self.node(source_name)
        sink = self.node(sink_name)
        total_flow = 0.0
        total_cost = 0.0
        paths = 0
        while True:
            dist = [float("inf")] * len(self.graph)
            parent: list[tuple[int, int] | None] = [None] * len(self.graph)
            dist[source] = 0.0
            heap = [(0.0, source)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for i, edge in enumerate(self.graph[u]):
                    v, cap, cost, _ = edge
                    if cap > 1e-9 and d + cost < dist[v]:
                        dist[v] = d + cost
                        parent[v] = (u, i)
                        heapq.heappush(heap, (dist[v], v))
            if parent[sink] is None:
                break
            bottleneck = float("inf")
            node = sink
            while node != source:
                u, i = parent[node]
                bottleneck = min(bottleneck, self.graph[u][i][1])
                node = u
            node = sink
            while node != source:
                u, i = parent[node]
                edge = self.graph[u][i]
                edge[1] -= bottleneck
                self.graph[edge[0]][int(edge[3])][1] += bottleneck
                total_cost += bottleneck * edge[2]
                node = u
            total_flow += bottleneck
            paths += 1
        return {"flow": total_flow, "cost": round(total_cost, 2), "augmenting_paths": paths}

    def flows(self) -> list[dict[str, Any]]:
        rows = []
        for u, edges in enumerate(self.graph):
            source = self.names[u]
            if not source.startswith("SKU:"):
                continue
            for v, _, cost, rev in edges:
                target = self.names[v]
                flow = self.graph[v][int(rev)][1]
                if target.startswith("SUPPLIER:") and flow > 1e-9:
                    rows.append({"product_id": source.replace("SKU:", ""), "supplier": target.replace("SUPPLIER:", ""), "units": round(flow, 2), "unit_cost": cost, "subtotal": round(flow * cost, 2)})
        return rows


def optimize_purchase_flow(options: pd.DataFrame, order: dict[str, float]) -> dict[str, Any]:
    network = MinCostFlow()
    requested = 0.0
    for product_id, quantity in order.items():
        requested += float(quantity)
        network.add_edge("SOURCE", f"SKU:{product_id}", float(quantity), 0.0)
    suppliers_seen: set[str] = set()
    for row in options.itertuples(index=False):
        product_id = str(row.product_id)
        if product_id not in order:
            continue
        network.add_edge(f"SKU:{product_id}", f"SUPPLIER:{row.supplier}", float(row.capacity_units), float(row.unit_cost))
        supplier = str(row.supplier)
        if supplier not in suppliers_seen:
            suppliers_seen.add(supplier)
            network.add_edge(f"SUPPLIER:{supplier}", "SINK", float(row.supplier_capacity), 0.0)
    solution = network.solve("SOURCE", "SINK")
    return {
        "method": "min_cost_flow",
        "plan": network.flows(),
        "total_cost": solution["cost"],
        "units_assigned": solution["flow"],
        "units_unfilled": round(requested - solution["flow"], 2),
        "augmenting_paths": solution["augmenting_paths"],
    }
