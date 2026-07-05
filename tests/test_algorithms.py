from __future__ import annotations

import pandas as pd

from app.core.algorithms import UFDS, families_from_projection


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
