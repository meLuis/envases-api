from __future__ import annotations

import pandas as pd

from app.core.algorithms import (
    UFDS,
    bellman_ford_savings,
    families_from_projection,
    knapsack_supply_budget,
    min_cost_flow_supply,
)
from app.core.graphs import bidirectional_bfs_path


def test_ufds_groups_projection_edges() -> None:
    edges = pd.DataFrame(
        [
            {"source": "A", "target": "B", "source_name": "Frasco A", "target_name": "Frasco B", "similarity": 0.8},
            {"source": "C", "target": "D", "source_name": "Tapa C", "target_name": "Tapa D", "similarity": 0.7},
        ]
    )
    families = families_from_projection(edges)
    assert set(families["family_size"]) == {2}


def test_ufds_kruskal_skips_cycle_edges() -> None:
    # Kruskal usa UFDS: al recorrer aristas ordenadas por peso, une componentes y
    # descarta las que cerrarian un ciclo. Sobre el triangulo A-B-C, la 3.a arista
    # (que reconecta A y C) debe rechazarse.
    edges = [("A", "B", 1.0), ("B", "C", 2.0), ("A", "C", 3.0)]
    uf = UFDS()
    accepted = []
    rejected = []
    for s, t, _ in sorted(edges, key=lambda e: e[2]):
        if uf.find(s) != uf.find(t):
            uf.union(s, t)
            accepted.append((s, t))
        else:
            rejected.append((s, t))
    assert accepted == [("A", "B"), ("B", "C")]
    assert rejected == [("A", "C")]
    # Todo quedo en un solo arbol/componente.
    assert len({uf.find(n) for n in ("A", "B", "C")}) == 1


def test_bidirectional_bfs_finds_client_to_supplier_path() -> None:
    # Cadena CLIENT:A — PRODUCT:P — SUPPLIER:S. El BFS bidireccional (dos frentes)
    # debe reconstruir el camino completo entre cliente y proveedor.
    edges = pd.DataFrame(
        [
            {"source": "CLIENT:A", "target": "PRODUCT:P"},
            {"source": "SUPPLIER:S", "target": "PRODUCT:P"},
        ]
    )
    path = bidirectional_bfs_path(edges, "CLIENT:A", "SUPPLIER:S")
    assert path == ["CLIENT:A", "PRODUCT:P", "SUPPLIER:S"]


def test_bellman_ford_marks_savings_as_negative_edges() -> None:
    # Un producto ofertado por 4 proveedores: costos 2.00 / 2.40 / 1.60 / 1.80.
    # La mediana es 1.90, así que 1.60 y 1.80 quedan por debajo → son ahorros
    # (aristas negativas). El mejor camino debe ser el proveedor más barato (1.60).
    options = pd.DataFrame(
        [
            {"product_id": "P1", "product_name": "Frasco", "supplier_id": "S1", "supplier": "Prov 1", "unit_cost": 2.00},
            {"product_id": "P1", "product_name": "Frasco", "supplier_id": "S2", "supplier": "Prov 2", "unit_cost": 2.40},
            {"product_id": "P1", "product_name": "Frasco", "supplier_id": "S3", "supplier": "Prov 3", "unit_cost": 1.60},
            {"product_id": "P1", "product_name": "Frasco", "supplier_id": "S4", "supplier": "Prov 4", "unit_cost": 1.80},
        ]
    )
    result = bellman_ford_savings(options)
    assert result["summary"]["negative_edges"] == 2
    best = result["best_paths"]
    assert not best.empty
    assert str(best.iloc[0]["supplier"]) == "Prov 3"
    assert float(best.iloc[0]["unit_cost"]) == 1.60


def test_knapsack_respects_budget() -> None:
    # Knapsack 0/1 sobre lotes proveedor-producto: nunca debe exceder el presupuesto
    # y debe elegir dentro de la capacidad disponible.
    options = pd.DataFrame(
        [
            {"product_id": "P1", "supplier": "Prov 1", "unit_cost": 2.0, "capacity_units": 10, "supplier_capacity": 10},
            {"product_id": "P2", "supplier": "Prov 2", "unit_cost": 5.0, "capacity_units": 10, "supplier_capacity": 10},
        ]
    )
    items = [
        {"product_id": "P1", "quantity": 5, "value": 5},
        {"product_id": "P2", "quantity": 5, "value": 5},
    ]
    result = knapsack_supply_budget(options, items, budget=20.0)
    assert result["total_cost"] <= 20.0
    assert result["total_units"] > 0
    assert result["budget_left"] >= 0


def test_min_cost_flow_respects_capacity_and_minimizes_cost() -> None:
    # El proveedor barato (2/u) solo cubre 5 unidades; el resto (3) va al caro
    # (5/u). Demanda 8 ⇒ costo minimo 5*2 + 3*5 = 25, respetando capacidades.
    options = pd.DataFrame(
        [
            {"product_id": "P1", "supplier": "CHEAP", "unit_cost": 2.0, "capacity_units": 5, "supplier_capacity": 100},
            {"product_id": "P1", "supplier": "PRICEY", "unit_cost": 5.0, "capacity_units": 100, "supplier_capacity": 100},
        ]
    )
    result = min_cost_flow_supply(options, [{"product_id": "P1", "quantity": 8}])
    assert result["ok"]
    assert result["served_units"] == 8
    assert result["total_cost"] == 25.0
    cheap = next(row for row in result["assignment"] if row["supplier"] == "CHEAP")
    assert cheap["units"] == 5.0
