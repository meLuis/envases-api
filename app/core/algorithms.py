from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, timedelta
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


def knapsack_supply_budget(
    options: pd.DataFrame,
    items: list[dict[str, Any]],
    budget: float,
    products: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """0/1 knapsack sobre ofertas/lotes de proveedores.

    Cada item DP es una oferta indivisible: comprar ese lote o no comprarlo.
    Peso = costo del lote. Valor = ganancia esperada al revenderlo.
    """
    requested = [str(item["product_id"]) for item in items if str(item.get("product_id", "")).strip()]
    requested = list(dict.fromkeys(requested))
    suggested_qty = {
        str(item["product_id"]): int(max(float(item.get("quantity", 0) or 0), 0))
        for item in items
        if str(item.get("product_id", "")).strip()
    }
    if products is not None and not products.empty:
        product_frame = products.copy()
        product_frame["product_id"] = product_frame["product_id"].astype(str)
        product_lookup = product_frame.set_index("product_id").to_dict("index")
    else:
        product_lookup = {}

    def product_meta(product_id: str) -> dict[str, Any]:
        row = product_lookup.get(product_id, {})
        return {
            "name": str(row.get("product_name") or product_id),
            "sale_price": float(row.get("unit_price_sale") or 0),
        }

    def stable_int(*parts: object) -> int:
        text = "|".join(str(part) for part in parts)
        return sum((index + 1) * ord(char) for index, char in enumerate(text))

    supplier_pool = [
        "Frascos Chorrillos SAC",
        "Andina Tapas SRL",
        "Nexo Industrial EIRL",
        "Altura PET SAC",
        "Atomizadores Lince SAC",
        "Nova Envases EIRL",
        "Plastinka Mayorista SAC",
        "Kuntur Packaging SAC",
    ]
    generic_supplier_names = {
        "",
        "QUESITO SA",
        "QUESITO S.A.",
        "COTIZACION ALTERNATIVA",
        "COTIZACIÓN ALTERNATIVA",
        "PROVEEDOR GENERICO",
        "PROVEEDOR GENÉRICO",
    }

    def readable_supplier_name(name: str) -> str:
        words = name.strip().split()
        fixed = []
        for word in words:
            upper = word.upper()
            if upper in {"SAC", "EIRL", "SRL", "SA", "S.A."}:
                fixed.append(upper.replace("S.A.", "SA"))
            else:
                fixed.append(word.capitalize())
        return " ".join(fixed)

    def supplier_name(product_id: str, raw_name: str, salt: object = 0, avoid: str | None = None) -> str:
        cleaned = str(raw_name or "").strip()
        if cleaned.upper() not in generic_supplier_names:
            return readable_supplier_name(cleaned)
        start = stable_int(product_id, cleaned, salt) % len(supplier_pool)
        for offset in range(len(supplier_pool)):
            candidate = supplier_pool[(start + offset) % len(supplier_pool)]
            if candidate != avoid:
                return candidate
        return supplier_pool[start]

    def package_label(units: int) -> str:
        if units >= 1000:
            return f"{units / 1000:.1f} millares"
        if units >= 100:
            return f"{units // 100} cientos"
        return f"{units} unidades"

    def product_options_for(product_id: str, minimum_units: int = 0) -> list[dict[str, Any]]:
        product_options = options.loc[options["product_id"].astype(str) == product_id].copy()
        meta = product_meta(product_id)
        rows: list[dict[str, Any]] = []
        if not product_options.empty:
            product_options["unit_cost"] = pd.to_numeric(product_options["unit_cost"], errors="coerce").fillna(0)
            product_options["capacity_units"] = pd.to_numeric(product_options["capacity_units"], errors="coerce").fillna(0)
            product_options = product_options.loc[(product_options["unit_cost"] > 0) & (product_options["capacity_units"] > 0)]
            for row in product_options.itertuples(index=False):
                supplier = supplier_name(product_id, str(getattr(row, "supplier", "") or ""), getattr(row, "unit_cost", 0))
                rows.append(
                    {
                        "supplier": supplier,
                        "unit_cost": float(getattr(row, "unit_cost", 0) or 0),
                        "capacity_units": int(max(float(getattr(row, "capacity_units", 0) or 0), minimum_units, 1)),
                        "synthetic": False,
                    }
                )

        if not rows:
            base_cost = meta["sale_price"] * 0.65 if meta["sale_price"] > 0 else 1.0
            first_supplier = supplier_name(product_id, "", "fallback-a")
            second_supplier = supplier_name(product_id, "", "fallback-b", avoid=first_supplier)
            rows = [
                {"supplier": first_supplier, "unit_cost": base_cost, "capacity_units": max(minimum_units, 1000, 1), "synthetic": True},
                {"supplier": second_supplier, "unit_cost": base_cost * 1.08, "capacity_units": max(minimum_units, 1000, 1), "synthetic": True},
            ]
        elif len(rows) == 1:
            only = rows[0]
            rows.append(
                {
                    "supplier": supplier_name(product_id, "Cotizacion alternativa", float(only["unit_cost"]), avoid=str(only["supplier"])),
                    "unit_cost": float(only["unit_cost"]) * 1.09,
                    "capacity_units": max(int(only["capacity_units"]), minimum_units, 1),
                    "synthetic": True,
                }
            )
        return sorted(rows, key=lambda item: (float(item["unit_cost"]), str(item["supplier"])))[:2]

    def expected_margin(product_id: str, supplier: str, profile: int, scaled: bool) -> float:
        if scaled:
            margin_seed = stable_int(product_id, supplier, "margin") % 7
            return [0.50, 0.43, 0.35, 0.27, 0.23, 0.18][profile] + margin_seed / 100
        margin_seed = stable_int(product_id, supplier, "margin")
        return 0.18 + (margin_seed % 24) / 100

    def make_lot(product_id: str, supplier: str, unit_cost: float, units: int, phase: str, selected: bool = False) -> dict[str, Any]:
        meta = product_meta(product_id)
        profile = stable_int(product_id, supplier, "knapsack_lot") % 6
        margin = expected_margin(product_id, supplier, profile, budget >= 1500)
        cost = unit_cost * units
        revenue = cost * (1 + margin)
        profit = max(revenue - cost, 0.0)
        return {
            "product_id": product_id,
            "product_name": meta["name"],
            "supplier": supplier,
            "units": int(units),
            "package_label": package_label(int(units)),
            "unit_cost": unit_cost,
            "sale_price": revenue / max(units, 1),
            "cost": cost,
            "revenue": revenue,
            "value": profit,
            "roi": profit / cost if cost else 0.0,
            "phase": phase,
            "selected": selected,
        }

    def view_lot(index: int, lot: dict[str, Any], selected: bool) -> dict[str, Any]:
        return {
            "offer_id": f"O{index + 1:02d}",
            "product_id": lot["product_id"],
            "product_name": lot["product_name"],
            "supplier": lot["supplier"],
            "units": int(lot["units"]),
            "package_label": str(lot.get("package_label", "")),
            "unit_cost": round(float(lot["unit_cost"]), 4),
            "sale_price": round(float(lot["sale_price"]), 4),
            "cost": round(float(lot["cost"]), 2),
            "revenue": round(float(lot["revenue"]), 2),
            "value": round(float(lot["value"]), 2),
            "roi": round(float(lot["roi"]), 4),
            "phase": str(lot.get("phase", "candidate")),
            "selected": selected,
        }

    lots: list[dict[str, Any]] = []
    for product_id in requested:
        product_options = product_options_for(product_id)
        for option in product_options:
            unit_cost = float(option["unit_cost"])
            supplier = str(option["supplier"])
            historical_units = int(max(float(option["capacity_units"] or 0), 1))
            profile = stable_int(product_id, supplier, "knapsack_lot") % 6
            if suggested_qty.get(product_id, 0) > 0:
                units = max(1, suggested_qty[product_id])
            elif budget >= 1500:
                budget_share = [0.22, 0.31, 0.42, 0.55, 0.68, 0.82][profile]
                target_cost = max(450.0, budget * budget_share)
                units = max(1, int(round(target_cost / max(unit_cost, 0.01))))
            else:
                units = historical_units
            lot = make_lot(product_id, supplier, unit_cost, units, "candidate")
            if lot["cost"] > 0 and lot["value"] > 0:
                lots.append(lot)

    capacity = int(round(max(budget, 0) * 100))
    if not lots or capacity <= 0:
        return {
            "method": "knapsack_dp",
            "plan": [],
            "items": [],
            "total_units": 0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
            "budget_left": round(budget, 2),
            "score": 0.0,
            "score_label": "ganancia_esperada",
            "status": "empty",
            "offers_considered": 0,
            "offers_selected": 0,
        }

    # Orden estable: primero ofertas con costo razonable y luego mayor ganancia.
    lots = sorted(lots, key=lambda lot: (lot["cost"], -lot["value"], lot["supplier"]))[:12]
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

    chosen_indices: list[int] = []
    w = capacity
    for i in range(len(lots) - 1, -1, -1):
        if w >= 0 and keep[i][w]:
            chosen_indices.append(i)
            w -= costs[i]
    chosen_indices.reverse()
    chosen = [lots[i] for i in chosen_indices]
    selected = set(chosen_indices)
    total_cost = sum(item["cost"] for item in chosen)
    total_revenue = sum(item["revenue"] for item in chosen)
    total_profit = sum(item["value"] for item in chosen)

    items_view = [view_lot(i, lot, i in selected) for i, lot in enumerate(lots)]
    optional_plan = [item for item in items_view if item["selected"]]
    return {
        "method": "knapsack_dp",
        "plan": optional_plan,
        "items": items_view,
        "optional_plan": optional_plan,
        "total_units": int(sum(item["units"] for item in chosen)),
        "total_cost": round(total_cost, 2),
        "total_revenue": round(total_revenue, 2),
        "budget_left": round(budget - total_cost, 2),
        "score": round(total_profit, 2),
        "score_label": "ganancia_esperada",
        "status": "ok",
        "offers_considered": len(lots),
        "offers_selected": len(chosen),
    }


def min_cost_flow_supply(options: pd.DataFrame, demand: list[dict[str, Any]]) -> dict[str, Any]:
    """Asignación de demanda a proveedores con costo y capacidad (Min-cost flow).

    Red: SOURCE → producto (cap = demanda) → (producto, proveedor) (cap =
    capacidad del proveedor para ese producto, costo = costo unitario) →
    proveedor (cap = capacidad total del proveedor) → SINK. Resuelve con
    caminos de costo mínimo sucesivos (SPFA/Bellman-Ford), garantizando el
    costo total mínimo para el flujo servido. Respeta demanda y capacidad.

    Además de la asignación final, expone:
      - `network`: TODAS las aristas del grafo en capas (con su capacidad),
        hayan recibido flujo o no, para poder dibujar la red completa
        (SOURCE/SINK incluidos) en vez de solo el resultado ya aplanado.
      - `steps`: una fila por camino aumentante encontrado (una "ronda" real
        del algoritmo), incluyendo reasignaciones cuando el flujo de una
        ronda libera capacidad usada por una ronda anterior (arista residual)
        para dársela a otro producto que comparte proveedor.
    """
    requested = {str(item["product_id"]): int(max(float(item.get("quantity", 0)), 0)) for item in demand}
    requested = {pid: qty for pid, qty in requested.items() if qty > 0}
    if options.empty or not requested:
        return {
            "ok": False,
            "reason": "empty",
            "assignment": [],
            "steps": [],
            "network": {"source_edges": [], "flow_edges": [], "sink_edges": []},
            "total_cost": 0.0,
            "served_units": 0,
            "demand_units": sum(requested.values()),
            "unmet": [],
        }

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

    def add_edge(u: int, v: int, cap: float, cost: float) -> int:
        fwd = len(edges)
        adj[u].append(fwd)
        edges.append([v, cap, cost, 0.0])
        adj[v].append(len(edges))
        edges.append([u, 0.0, -cost, 0.0])
        return fwd

    # SOURCE → producto (capacidad = demanda).
    source_edge_of: dict[str, int] = {}
    for pid, qty in requested.items():
        source_edge_of[pid] = add_edge(SOURCE, nid(f"P:{pid}"), float(qty), 0.0)

    # producto → (producto, proveedor) con costo unitario; y proveedor → SINK
    # con la capacidad total del proveedor.
    supplier_cap: dict[str, float] = {}
    flow_edge_of: dict[tuple[str, str], int] = {}
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
        flow_edge_of[(pid, supplier)] = add_edge(nid(f"P:{pid}"), ps_node, cap, cost)
        add_edge(ps_node, nid(f"S:{supplier}"), cap, 0.0)
        supplier_cap[supplier] = max(supplier_cap.get(supplier, 0.0), float(getattr(row, "supplier_capacity", 0) or 0) or cap)

    sink_edge_of: dict[str, int] = {}
    for supplier, cap in supplier_cap.items():
        sink_edge_of[supplier] = add_edge(nid(f"S:{supplier}"), SINK, cap, 0.0)

    total_nodes = len(node_id)
    rev_index = {v: k for k, v in node_id.items()}

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
    steps: list[dict[str, Any]] = []
    step_no = 0
    while True:
        dist, prev_edge = spfa()
        if dist[SINK] == float("inf"):
            break
        # Cuello de botella del camino + snapshot (cap, flujo) de cada arista
        # ANTES de aplicar esta ronda, en orden SOURCE → SINK.
        bottleneck = float("inf")
        path_edges: list[int] = []
        v = SINK
        while v != SOURCE:
            eidx = prev_edge[v]
            cap, flow = edges[eidx][1], edges[eidx][3]
            bottleneck = min(bottleneck, cap - flow)
            path_edges.append(eidx)
            v = int(edges[eidx ^ 1][0])
        path_edges.reverse()
        snapshot = {eidx: (edges[eidx][1], edges[eidx][3]) for eidx in path_edges}

        for eidx in path_edges:
            edges[eidx][3] += bottleneck
            edges[eidx ^ 1][3] -= bottleneck
        served += bottleneck
        step_no += 1

        # Cada arista producto↔proveedor tocada por este camino es una fila:
        # signo + = nueva asignación, signo − = reasignación (se libera
        # capacidad usada en una ronda anterior para dársela a otro camino
        # más barato en la misma ronda).
        for eidx in path_edges:
            u = int(edges[eidx ^ 1][0])
            v_ = int(edges[eidx][0])
            u_name = rev_index.get(u, "")
            v_name = rev_index.get(v_, "")
            cap_before, flow_before = snapshot[eidx]
            if u_name.startswith("P:") and v_name.startswith("PS:"):
                _, pid, supplier = v_name.split(":", 2)
                unit_cost = edges[eidx][2]
                delta = bottleneck
            elif u_name.startswith("PS:") and v_name.startswith("P:"):
                _, pid, supplier = u_name.split(":", 2)
                unit_cost = -edges[eidx][2]
                delta = -bottleneck
            else:
                continue
            step_cost = delta * unit_cost
            total_cost += step_cost
            steps.append(
                {
                    "step": step_no,
                    "product_id": pid,
                    "supplier": supplier,
                    "delta_units": round(delta, 2),
                    "unit_cost": round(unit_cost, 4),
                    "capacity_before": round(cap_before - flow_before, 2),
                    "step_cost": round(step_cost, 2),
                    "cumulative_cost": round(total_cost, 2),
                    "served_units": round(edges[source_edge_of[pid]][3], 2) if pid in source_edge_of else None,
                    "demand_remaining": round(requested.get(pid, 0) - edges[source_edge_of[pid]][3], 2) if pid in source_edge_of else None,
                }
            )

    # Asignación final agregada por (producto, proveedor): la vista "resultado".
    assignment = []
    for (pid, supplier), eidx in flow_edge_of.items():
        flow = edges[eidx][3]
        if flow <= 1e-9:
            continue
        cost = edges[eidx][2]
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

    # Red completa (todas las opciones, tengan o no flujo) para dibujar SOURCE,
    # SINK y las capacidades reales, no solo las aristas ganadoras.
    source_edges = [
        {
            "product_id": pid,
            "capacity": edges[eidx][1],
            "flow": round(edges[eidx][3], 2),
        }
        for pid, eidx in source_edge_of.items()
    ]
    flow_edges = [
        {
            "product_id": pid,
            "supplier": supplier,
            "capacity": edges[eidx][1],
            "unit_cost": round(edges[eidx][2], 4),
            "flow": round(edges[eidx][3], 2),
        }
        for (pid, supplier), eidx in flow_edge_of.items()
    ]
    sink_edges = [
        {
            "supplier": supplier,
            "capacity": edges[eidx][1],
            "flow": round(edges[eidx][3], 2),
        }
        for supplier, eidx in sink_edge_of.items()
    ]

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
        "steps": steps,
        "network": {"source_edges": source_edges, "flow_edges": flow_edges, "sink_edges": sink_edges},
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


def bellman_ford_campaigns(
    options: pd.DataFrame,
    products: pd.DataFrame,
    today: date | None = None,
    max_offers: int = 180,
) -> dict[str, Any]:
    """Bellman-Ford sobre campañas sintéticas.

    A partir del historial de compras se generan alternativas de compra:
    comprar hoy o esperar una campaña/cupón cercano. Los costos positivos son
    compra/espera; los pesos negativos son descuentos. El grafo es acíclico por
    construcción, así que los descuentos no pueden aplicarse infinitamente.
    """
    if options.empty:
        return {
            "candidates": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "best_paths": pd.DataFrame(),
            "summary": {"status": "empty"},
        }

    current = today or date.today()
    data = options.copy()
    for column in ("unit_cost", "capacity_units"):
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data = data.loc[(data["unit_cost"] > 0) & (data["capacity_units"] > 0)].copy()
    if data.empty:
        return {
            "candidates": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "best_paths": pd.DataFrame(),
            "summary": {"status": "empty"},
        }

    product_frame = products.copy() if products is not None and not products.empty else pd.DataFrame()
    if not product_frame.empty:
        product_frame["product_id"] = product_frame["product_id"].astype(str)
        product_lookup = product_frame.set_index("product_id").to_dict("index")
    else:
        product_lookup = {}

    data["product_id"] = data["product_id"].astype(str)
    data = (
        data.sort_values(["product_id", "unit_cost", "capacity_units"], ascending=[True, True, False])
        .groupby("product_id", as_index=False)
        .head(3)
        .head(max_offers)
    )

    edges: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    def stable_int(*parts: object) -> int:
        text = "|".join(str(part) for part in parts)
        return sum((index + 1) * ord(char) for index, char in enumerate(text))

    def add_edge(source: str, target: str, weight: float, edge_type: str, row: dict[str, Any]) -> None:
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": round(weight, 4),
                "edge_type": edge_type,
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "supplier": row["supplier"],
                "label": row.get("label", edge_type),
            }
        )

    campaign_names = ["Cyberday", "Aniversario proveedor", "Cierre de mes"]
    for idx, row in enumerate(data.itertuples(index=False), start=1):
        product_id = str(row.product_id)
        supplier = str(getattr(row, "supplier", "") or "")
        product_name = str(getattr(row, "product_name", "") or product_id)
        product_meta = product_lookup.get(product_id, {})
        sale_price = float(product_meta.get("unit_price_sale") or 0)
        unit_cost = float(getattr(row, "unit_cost", 0) or 0)
        historical_units = int(max(float(getattr(row, "capacity_units", 0) or 0), 1))
        target_cost = 1500 + (stable_int(product_id, supplier, "lot") % 3001)
        units = max(historical_units, int(round(target_cost / max(unit_cost, 0.01))))
        if sale_price <= unit_cost:
            sale_price = unit_cost * 1.35
        base_cost = unit_cost * units
        base_revenue = sale_price * units
        offer = f"OFFER:{idx:03d}"
        source_row = {
            "product_id": product_id,
            "product_name": product_name,
            "supplier": supplier,
            "label": "costo base",
        }
        add_edge("SOURCE", offer, base_cost, "base_cost", source_row)

        variants = [
            {
                "name": "Comprar hoy",
                "wait_days": 0,
                "campaign_pct": 0.0,
                "coupon": 0.0,
                "coupon_name": "",
            }
        ]
        seed = stable_int(product_id, supplier)
        for offset in range(2):
            wait_days = 2 + (seed + offset * 5) % 8
            campaign_pct = 0.03 + ((seed // (offset + 3)) % 10) / 100
            coupon = 0.0
            if base_cost >= 1500 and (seed + offset) % 3 != 0:
                coupon = 60.0 + ((seed // (offset + 5)) % 5) * 40.0
            variants.append(
                {
                    "name": campaign_names[(seed + offset) % len(campaign_names)],
                    "wait_days": wait_days,
                    "campaign_pct": campaign_pct,
                    "coupon": coupon,
                    "coupon_name": f"CUPON{10 + ((seed + offset) % 70)}" if coupon > 0 else "",
                }
            )

        for alt_idx, variant in enumerate(variants):
            wait_days = int(variant["wait_days"])
            campaign_pct = float(variant["campaign_pct"])
            coupon = float(variant["coupon"])
            campaign_discount = base_cost * campaign_pct
            wait_cost = wait_days * max(70.0, base_cost * 0.022)
            buy_date = current + timedelta(days=wait_days)
            decision = f"DECISION:{idx:03d}:{alt_idx}"
            campaign = f"CAMPAIGN:{idx:03d}:{alt_idx}"
            coupon_node = f"COUPON:{idx:03d}:{alt_idx}"
            final = f"FINAL:{idx:03d}:{alt_idx}"
            path = [offer, decision]

            add_edge(
                offer,
                decision,
                wait_cost,
                "wait_cost" if wait_days else "buy_today",
                {**source_row, "label": f"esperar {wait_days} dia(s)" if wait_days else "comprar hoy"},
            )
            if campaign_discount > 0:
                add_edge(
                    decision,
                    campaign,
                    -campaign_discount,
                    "campaign_discount",
                    {**source_row, "label": str(variant["name"])},
                )
                path.append(campaign)
                previous = campaign
            else:
                previous = decision
            if coupon > 0:
                add_edge(
                    previous,
                    coupon_node,
                    -coupon,
                    "coupon_discount",
                    {**source_row, "label": str(variant["coupon_name"])},
                )
                path.append(coupon_node)
                previous = coupon_node
            add_edge(previous, final, 0.0, "final", {**source_row, "label": "comprar"})
            path.append(final)

            final_cost = base_cost + wait_cost - campaign_discount - coupon
            savings = base_cost - final_cost
            candidates.append(
                {
                    "product_id": product_id,
                    "product_name": product_name,
                    "supplier": supplier,
                    "units": units,
                    "base_cost": round(base_cost, 2),
                    "base_revenue": round(base_revenue, 2),
                    "decision": "esperar" if wait_days else "comprar_hoy",
                    "campaign": str(variant["name"]),
                    "coupon": str(variant["coupon_name"]),
                    "buy_date": buy_date.isoformat(),
                    "wait_days": wait_days,
                    "wait_cost": round(wait_cost, 2),
                    "campaign_discount": round(campaign_discount, 2),
                    "coupon_discount": round(coupon, 2),
                    "final_cost": round(final_cost, 2),
                    "savings": round(savings, 2),
                    "savings_pct": round(savings / base_cost, 4) if base_cost else 0.0,
                    "final_node": final,
                    "path": "SOURCE|" + "|".join(path),
                }
            )

    edges_df = pd.DataFrame(edges)
    candidates_df = pd.DataFrame(candidates)
    if edges_df.empty or candidates_df.empty:
        return {
            "candidates": candidates_df,
            "edges": edges_df,
            "best_paths": pd.DataFrame(),
            "summary": {"status": "empty"},
        }

    nodes = sorted(set(edges_df["source"].astype(str)) | set(edges_df["target"].astype(str)))
    dist = {node: float("inf") for node in nodes}
    pred: dict[str, str | None] = {node: None for node in nodes}
    dist["SOURCE"] = 0.0
    edge_rows = edges_df.to_dict("records")
    iterations = 0
    for _ in range(max(len(nodes) - 1, 1)):
        iterations += 1
        changed = False
        for edge in edge_rows:
            source = str(edge["source"])
            target = str(edge["target"])
            weight = float(edge["weight"])
            if dist[source] < float("inf") and dist[source] + weight < dist[target]:
                dist[target] = dist[source] + weight
                pred[target] = source
                changed = True
        if not changed:
            break

    best_rows = []
    for _, group in candidates_df.groupby(["product_id", "supplier"], as_index=False):
        ranked = group.sort_values(["final_cost", "wait_days"], ascending=[True, True])
        best_rows.append(ranked.iloc[0].to_dict())
    best_paths = pd.DataFrame(best_rows).sort_values(["savings", "savings_pct"], ascending=[False, False])
    negative_edges = int((edges_df["weight"] < 0).sum())
    wait_wins = int((best_paths["decision"] == "esperar").sum()) if not best_paths.empty else 0

    return {
        "candidates": candidates_df,
        "edges": edges_df,
        "best_paths": best_paths,
        "summary": {
            "status": "ok",
            "algorithm": "Bellman-Ford",
            "model": "synthetic_campaigns",
            "current_date": current.isoformat(),
            "vertices": len(nodes),
            "edges": len(edges_df),
            "iterations": iterations,
            "negative_edges": negative_edges,
            "offers": int(data.shape[0]),
            "alternatives": int(candidates_df.shape[0]),
            "wait_wins": wait_wins,
        },
    }
