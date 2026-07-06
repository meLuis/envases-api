from __future__ import annotations

import pandas as pd

from app.core.algorithms import UFDS, families_from_projection, min_cost_flow_supply
from app.core.graphs import a_star_route, tarjan_critical


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


def test_a_star_picks_lower_cost_route_with_admissible_heuristic() -> None:
    # A → C directo (25 km) vs A → B → C (1.5 + 18 = 19.5 km). Con h(n) admisible
    # (distancia recta al destino) A* debe devolver la ruta de menor costo total.
    coords = {
        "CLIENT:A": (-12.05, -77.05),
        "SUPPLIER:B": (-12.06, -77.04),
        "SUPPLIER:C": (-12.20, -76.90),
    }
    edges = pd.DataFrame(
        [
            {"source": "CLIENT:A", "target": "SUPPLIER:B", "km": 1.5},
            {"source": "SUPPLIER:B", "target": "SUPPLIER:C", "km": 18.0},
            {"source": "CLIENT:A", "target": "SUPPLIER:C", "km": 25.0},
        ]
    )
    result = a_star_route(edges, coords, "CLIENT:A", "SUPPLIER:C")
    assert result["ok"]
    assert result["path"] == ["CLIENT:A", "SUPPLIER:B", "SUPPLIER:C"]
    assert result["total_km"] == 19.5


def test_tarjan_detects_seeded_bridge_node() -> None:
    # Dos triangulos (a-b-c) y (d-e-f) unidos por el nodo X: X es puente cuya
    # caida fragmenta la red en dos componentes.
    edges = pd.DataFrame(
        [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
            {"source": "c", "target": "a"},
            {"source": "c", "target": "X"},
            {"source": "X", "target": "d"},
            {"source": "d", "target": "e"},
            {"source": "e", "target": "f"},
            {"source": "f", "target": "d"},
        ]
    )
    result = tarjan_critical(edges)
    critical = {item["node"] for item in result["articulation"]}
    assert "X" in critical
    # Al remover X quedan al menos 2 componentes.
    x_impact = next(item for item in result["articulation"] if item["node"] == "X")
    assert x_impact["components_after_removal"] >= 2


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
