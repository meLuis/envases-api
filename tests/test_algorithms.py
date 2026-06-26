from __future__ import annotations

import pandas as pd

from app.core.algorithms import families_from_projection, optimize_purchase_flow


def test_ufds_groups_projection_edges() -> None:
    edges = pd.DataFrame(
        [
            {"source": "A", "target": "B", "source_name": "Frasco A", "target_name": "Frasco B", "similarity": 0.8},
            {"source": "C", "target": "D", "source_name": "Tapa C", "target_name": "Tapa D", "similarity": 0.7},
        ]
    )
    families = families_from_projection(edges)
    assert set(families["family_size"]) == {2}


def test_min_cost_flow_assigns_cheapest_available_supplier() -> None:
    options = pd.DataFrame(
        [
            {"product_id": "P1", "supplier": "A", "unit_cost": 1.0, "capacity_units": 5, "supplier_capacity": 5},
            {"product_id": "P1", "supplier": "B", "unit_cost": 2.0, "capacity_units": 10, "supplier_capacity": 10},
        ]
    )
    result = optimize_purchase_flow(options, {"P1": 8})
    assert result["units_assigned"] == 8
    assert result["total_cost"] == 11.0

