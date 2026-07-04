from __future__ import annotations

import pandas as pd

from app.core.semantic_search import SemanticSearchIndex


def _index() -> SemanticSearchIndex:
    nodes = pd.DataFrame(
        [
            {"node_id": "PRODUCT:P1", "node_type": "PRODUCT", "label": "FRASCO VIDRIO AMBAR 100ML"},
            {"node_id": "PRODUCT:P2", "node_type": "PRODUCT", "label": "FRASCO PLASTICO AMBAR 100ML"},
            {"node_id": "PRODUCT:P3", "node_type": "PRODUCT", "label": "FRASCO VIDRIO AMBAR 120ML"},
            {"node_id": "TYPE:FRASCO", "node_type": "TYPE", "label": "frasco"},
            {"node_id": "MATERIAL:VIDRIO", "node_type": "MATERIAL", "label": "VIDRIO"},
            {"node_id": "MATERIAL:PLASTICO", "node_type": "MATERIAL", "label": "PLASTICO"},
            {"node_id": "COLOR:AMBAR", "node_type": "COLOR", "label": "AMBAR"},
            {"node_id": "CAPACITY:100ML", "node_type": "CAPACITY", "label": "100ML"},
            {"node_id": "CAPACITY:120ML", "node_type": "CAPACITY", "label": "120ML"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "PRODUCT:P1", "target": "TYPE:FRASCO", "weight": 0.93},
            {"source": "PRODUCT:P1", "target": "MATERIAL:VIDRIO", "weight": 0.94},
            {"source": "PRODUCT:P1", "target": "COLOR:AMBAR", "weight": 0.94},
            {"source": "PRODUCT:P1", "target": "CAPACITY:100ML", "weight": 0.95},
            {"source": "PRODUCT:P2", "target": "TYPE:FRASCO", "weight": 0.93},
            {"source": "PRODUCT:P2", "target": "MATERIAL:PLASTICO", "weight": 0.94},
            {"source": "PRODUCT:P2", "target": "COLOR:AMBAR", "weight": 0.94},
            {"source": "PRODUCT:P2", "target": "CAPACITY:100ML", "weight": 0.95},
            {"source": "PRODUCT:P3", "target": "TYPE:FRASCO", "weight": 0.93},
            {"source": "PRODUCT:P3", "target": "MATERIAL:VIDRIO", "weight": 0.94},
            {"source": "PRODUCT:P3", "target": "COLOR:AMBAR", "weight": 0.94},
            {"source": "PRODUCT:P3", "target": "CAPACITY:120ML", "weight": 0.95},
        ]
    )
    return SemanticSearchIndex.from_frames(nodes, edges)


def test_semantic_search_requires_all_resolved_concepts() -> None:
    results = _index().search("frasco vidrio ambar 100ml", k=10)

    assert [item["product"] for item in results] == ["PRODUCT:P1"]
    assert results[0]["seed_coverage"] == results[0]["total_seeds"] == 4


def test_semantic_search_exact_missing_capacity_returns_no_results() -> None:
    index = _index()

    assert index.search("frasco vidrio ambar 123ml", k=10) == []
    assert index.last_stats["unresolved_exact_filters"] == ["capacity:123ML"]


def _index_with_sales() -> SemanticSearchIndex:
    # Dos frascos ambar 100ML que cumplen TODA la consulta; P2 vende más.
    nodes = pd.DataFrame(
        [
            {"node_id": "PRODUCT:P1", "node_type": "PRODUCT", "label": "FRASCO AMBAR 100ML A", "units_sold": 5},
            {"node_id": "PRODUCT:P2", "node_type": "PRODUCT", "label": "FRASCO AMBAR 100ML B", "units_sold": 42},
            {"node_id": "TYPE:FRASCO", "node_type": "TYPE", "label": "frasco", "units_sold": ""},
            {"node_id": "COLOR:AMBAR", "node_type": "COLOR", "label": "AMBAR", "units_sold": ""},
            {"node_id": "CAPACITY:100ML", "node_type": "CAPACITY", "label": "100ML", "units_sold": ""},
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "PRODUCT:P1", "target": "TYPE:FRASCO", "weight": 0.95},
            {"source": "PRODUCT:P1", "target": "COLOR:AMBAR", "weight": 0.95},
            {"source": "PRODUCT:P1", "target": "CAPACITY:100ML", "weight": 0.95},
            {"source": "PRODUCT:P2", "target": "TYPE:FRASCO", "weight": 0.90},
            {"source": "PRODUCT:P2", "target": "COLOR:AMBAR", "weight": 0.90},
            {"source": "PRODUCT:P2", "target": "CAPACITY:100ML", "weight": 0.90},
        ]
    )
    return SemanticSearchIndex.from_frames(nodes, edges)


def test_finalists_ordered_by_sales() -> None:
    results = _index_with_sales().search("frasco ambar 100ml", k=10)

    # Ambos cumplen todo; el más vendido (P2) va primero pese a menor score BFS.
    assert [item["product"] for item in results] == ["PRODUCT:P2", "PRODUCT:P1"]
    assert results[0]["units_sold"] == 42


def test_limit_returns_top_sold_finalist() -> None:
    results = _index_with_sales().search("frasco ambar 100ml", k=1)

    assert [item["product"] for item in results] == ["PRODUCT:P2"]
