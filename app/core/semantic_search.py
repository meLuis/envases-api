"""Búsqueda semántica sobre G_attr (BFS multi-semilla). Port del motor base.

El vocabulario se aprende del grafo: cada nodo atributo de G_attr es un término
reconocible, así el mismo buscador funciona para cualquier catálogo procesado.

Flujo:
    query -> tokens normalizados -> semillas (nodos atributo de G_attr)
          -> BFS O(V+E) con decaimiento por distancia
          -> filtro estricto por cobertura directa de conceptos
          -> boost por cobertura de semillas
          -> top-k

Reglas:
- Cada concepto resuelto de la consulta debe cumplirse: "frasco vidrio ambar"
  exige un producto conectado directamente a TYPE:FRASCO, MATERIAL:VIDRIO y
  COLOR:AMBAR. Si una palabra resuelve a varios nodos, basta uno de ese grupo.
- Atributos numéricos son conceptos exactos: 100ML solo devuelve productos de
  exactamente 100ML, sin aproximaciones.
- Cobertura: un producto que toca más conceptos multiplica su puntaje con
  ((cobertura/total)^2 * 20 + 1). Con filtro estricto, los resultados finales
  salen con cobertura total; el puntaje sigue ordenando dentro de esa intersección.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.attributes import extract_capacity, extract_mouth_size
from app.core.graphs import attr_node_id
from app.core.text import normalize_text


NUMERIC_NODE_TYPES = {"CAPACITY", "MOUTH_SIZE"}
DISTANCE_DECAY = (5, 4, 3, 2, 1)  # peso por distancia BFS 0..4+


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ES"):
        return token[:-2]
    if len(token) > 3 and token.endswith("S"):
        return token[:-1]
    return token


@dataclass
class SeedGroup:
    """Concepto resuelto de la consulta: un grupo OR de nodos atributo.

    Ejemplo: si "atomizador" existe como TYPE y SUBTYPE, ambos nodos entran en
    el mismo grupo. El filtro estricto exige al menos uno de ellos, no ambos.
    """

    key: str
    nodes: list[str]
    exact: bool = False


@dataclass
class SemanticSearchIndex:
    """Índice de búsqueda construido desde los CSV de G_attr."""

    adjacency: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    node_type: dict[str, str] = field(default_factory=dict)
    label_to_nodes: dict[str, list[str]] = field(default_factory=dict)
    product_labels: dict[str, str] = field(default_factory=dict)
    last_stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_frames(cls, nodes: pd.DataFrame, edges: pd.DataFrame) -> "SemanticSearchIndex":
        index = cls()
        for _, node in nodes.iterrows():
            nid = str(node["node_id"])
            ntype = str(node["node_type"])
            index.node_type[nid] = ntype
            label = normalize_text(node.get("label"))
            if ntype == "PRODUCT":
                index.product_labels[nid] = str(node.get("label", ""))
            elif label:
                index.label_to_nodes.setdefault(label, []).append(nid)
        for _, edge in edges.iterrows():
            source, target = str(edge["source"]), str(edge["target"])
            try:
                weight = float(edge["weight"])
            except (TypeError, ValueError):
                weight = 1.0
            index.adjacency.setdefault(source, []).append((target, weight))
            index.adjacency.setdefault(target, []).append((source, weight))
        return index

    # ── Semillas ────────────────────────────────────────────────────────────

    def extract_seed_groups(self, query: str) -> list[SeedGroup]:
        """Resuelve el query contra el vocabulario del grafo.

        Devuelve grupos de semillas por concepto de consulta.
        Un filtro numérico se registra aunque su nodo NO exista en el grafo:
        pedir 123ML cuando ningún producto tiene 123ML debe dar cero resultados.
        """
        text = normalize_text(query)
        groups: list[SeedGroup] = []
        seen_keys: set[str] = set()

        def add_group(key: str, candidate_nodes: list[str], exact: bool = False) -> None:
            if key in seen_keys:
                return
            nodes = [node for node in dict.fromkeys(candidate_nodes) if node in self.node_type]
            node_set = set(nodes)
            if node_set and any(set(group.nodes) == node_set for group in groups):
                seen_keys.add(key)
                return
            seen_keys.add(key)
            if nodes or exact:
                groups.append(SeedGroup(key=key, nodes=nodes, exact=exact))

        capacity_value, capacity_unit, _, _ = extract_capacity(text)
        if capacity_value is not None and capacity_unit:
            capacity_node = attr_node_id("CAPACITY", f"{capacity_value:g}{capacity_unit.upper()}")
            add_group(f"capacity:{capacity_value:g}{capacity_unit.upper()}", [capacity_node], exact=True)
        mouth_size, _, _ = extract_mouth_size(text)
        if mouth_size is not None:
            mouth_node = attr_node_id("MOUTH_SIZE", f"{mouth_size:g}MM")
            add_group(f"mouth_size:{mouth_size:g}MM", [mouth_node], exact=True)

        for token in text.split():
            matched_nodes: list[str] = []
            for candidate in (token, _singular(token)):
                matched_nodes.extend(self.label_to_nodes.get(candidate, []))
            add_group(f"term:{token}", matched_nodes)
        return groups

    def extract_seeds(self, query: str) -> tuple[list[str], list[str]]:
        """Compatibilidad para callers antiguos: devuelve semillas y exactos."""
        groups = self.extract_seed_groups(query)
        seeds = [node for group in groups for node in group.nodes]
        exact_filters = [node for group in groups if group.exact for node in group.nodes]
        return seeds, exact_filters

    # ── Búsqueda ──────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        seed_groups = self.extract_seed_groups(query)
        seeds = [node for group in seed_groups for node in group.nodes]
        unresolved_exact = [group.key for group in seed_groups if group.exact and not group.nodes]
        if not seeds:
            self.last_stats = {
                "seeds": [],
                "seed_groups": len(seed_groups),
                "unresolved_exact_filters": unresolved_exact,
                "expanded_nodes": 0,
            }
            return []

        total_groups = len(seed_groups)
        if unresolved_exact:
            self.last_stats = {
                "seeds": seeds,
                "seed_groups": total_groups,
                "exact_filters": sum(1 for group in seed_groups if group.exact),
                "unresolved_exact_filters": unresolved_exact,
                "expanded_nodes": 0,
                "strict_filter_applied": True,
            }
            return []

        # Cobertura directa por concepto: cuántos grupos de la consulta cumple
        # el producto como atributo directo. Dentro de un grupo la relación es OR.
        matched_groups: dict[str, set[str]] = {}
        for group in seed_groups:
            adjacent_products: set[str] = set()
            for seed in group.nodes:
                adjacent_products.update(
                    neighbor
                    for neighbor, _ in self.adjacency.get(seed, [])
                    if self.node_type.get(neighbor) == "PRODUCT"
                )
            for product in adjacent_products:
                matched_groups.setdefault(product, set()).add(group.key)
        coverage = {product: len(groups) for product, groups in matched_groups.items()}
        strict_allowed = {product for product, count in coverage.items() if count == total_groups}

        # BFS multi-semilla con decaimiento por distancia.
        visited: set[str] = set(seeds)
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        score: dict[str, float] = {}
        expanded = 0

        while queue:
            node, dist = queue.popleft()
            expanded += 1
            decay = DISTANCE_DECAY[min(dist, len(DISTANCE_DECAY) - 1)]
            for neighbor, weight in self.adjacency.get(node, []):
                if self.node_type.get(neighbor) == "PRODUCT":
                    score[neighbor] = score.get(neighbor, 0.0) + decay * weight
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        if not score:
            self.last_stats = {
                "seeds": seeds,
                "seed_groups": total_groups,
                "expanded_nodes": expanded,
                "strict_filter_applied": True,
            }
            return []

        for product in score:
            cov = coverage.get(product, 0)
            score[product] *= (cov / total_groups) ** 2 * 20 + 1

        results = []
        for product in sorted(score, key=score.get, reverse=True):
            if product not in strict_allowed:
                continue
            results.append(
                {
                    "product": product,
                    "label": self.product_labels.get(product, ""),
                    "relevance": round(score[product], 2),
                    "seed_coverage": coverage.get(product, 0),
                    "total_seeds": total_groups,
                }
            )
            if len(results) >= k:
                break

        self.last_stats = {
            "seeds": seeds,
            "seed_groups": total_groups,
            "exact_filters": sum(1 for group in seed_groups if group.exact),
            "expanded_nodes": expanded,
            "scored_products": len(score),
            "strict_candidates": len(strict_allowed),
            "strict_filter_applied": True,
        }
        return results
