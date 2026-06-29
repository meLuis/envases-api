"""Imágenes PNG estáticas para los grafos principales.

G_attr conserva los tres estilos del proyecto base:
1. Proyección en columnas por capa (TYPE, SUBTYPE, … MOUTH_SIZE).
2. Subgrafo force-directed producto ↔ atributos principales.
3. Familia de atributos (FRASCO + VIDRIO + AMBAR…).

Además se generan láminas estáticas completas para G_sales, G_purchases y
G_business. Las vistas completas priorizan evidenciar escala: pueden ser densas
si el grafo tiene miles de nodos, mientras que las muestras secundarias quedan
para lectura rápida.

Adaptado al esquema de aristas del backend actual (columna `edge_type` en vez de
`relation`; el peso de arista usa `weight`/`amount` cuando existen).
Se importa de forma perezosa para no exigir matplotlib/networkx salvo cuando se
generan imágenes.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


NODE_COLORS = {
    "PRODUCT": "#64748b",
    "CLIENT": "#c084fc",
    "SUPPLIER": "#fb7185",
    "DOCUMENT": "#facc15",
    "TYPE": "#3b82f6",
    "SUBTYPE": "#94a3b8",
    "ACCESSORY": "#14b8a6",
    "SHAPE": "#ec4899",
    "FEATURE": "#eab308",
    "MATERIAL": "#f97316",
    "COLOR": "#22c55e",
    "CAPACITY": "#a855f7",
    "MOUTH_SIZE": "#ef4444",
}

LAYER_ORDER = [
    "TYPE",
    "SUBTYPE",
    "ACCESSORY",
    "SHAPE",
    "FEATURE",
    "MATERIAL",
    "COLOR",
    "CAPACITY",
    "MOUTH_SIZE",
]


def _import_libs():
    import matplotlib

    matplotlib.use("Agg")  # backend headless (sin display, apto para servidor)
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    return mpatches, plt, nx


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def _safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def load_graph_tables(stage2_output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = Path(stage2_output_dir)
    nodes = _safe_read_csv(base / "semantic_attribute_graph_nodes.csv")
    edges = _safe_read_csv(base / "semantic_attribute_graph_edges.csv")
    metrics = _safe_read_json(base / "semantic_attribute_graph_metrics.json")
    return nodes, edges, metrics


def build_networkx_graph(nodes: pd.DataFrame, edges: pd.DataFrame, nx) -> Any:
    graph = nx.Graph()
    for _, node in nodes.iterrows():
        graph.add_node(
            node["node_id"],
            label=str(node.get("label", node["node_id"])),
            node_type=node["node_type"],
            sales_rows=0.0,
            purchases_rows=0.0,
        )
    for _, edge in edges.iterrows():
        weight = _to_float(edge.get("weight", 1), 1.0)
        graph.add_edge(
            edge["source"],
            edge["target"],
            weight=weight,
            amount=_to_float(edge.get("amount", 0), 0.0),
            edge_type=str(edge.get("edge_type", edge.get("relation", ""))),
        )
    return graph


def load_named_graph_tables(base_dir: str | Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = Path(base_dir)
    nodes = _safe_read_csv(base / f"transaction_graph_{name}_nodes.csv")
    edges = _safe_read_csv(base / f"transaction_graph_{name}_edges.csv")
    metrics = _safe_read_json(base / f"transaction_graph_{name}_metrics.json")
    return nodes, edges, metrics


def product_activity_score(graph, product_node: str) -> float:
    data = graph.nodes[product_node]
    return float(data.get("sales_rows", 0) or 0) + float(data.get("purchases_rows", 0) or 0)


def select_focus_product_attribute_graph(graph, top_attribute_count: int = 16, max_products: int = 120):
    attribute_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") != "PRODUCT"]
    top_attrs = sorted(attribute_nodes, key=lambda node: graph.degree(node), reverse=True)[:top_attribute_count]

    candidate_products = set()
    for attr in top_attrs:
        candidate_products.update(
            neighbor for neighbor in graph.neighbors(attr) if graph.nodes[neighbor].get("node_type") == "PRODUCT"
        )

    selected_products = sorted(
        candidate_products,
        key=lambda node: (product_activity_score(graph, node), graph.degree(node)),
        reverse=True,
    )[:max_products]
    selected = set(top_attrs) | set(selected_products)
    return graph.subgraph(selected).copy()


def build_attribute_projection(graph, max_attribute_nodes: int = 85, min_edge_weight: int = 3):
    import networkx as nx

    attribute_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") != "PRODUCT"]
    selected_attrs = set(
        sorted(attribute_nodes, key=lambda node: graph.degree(node), reverse=True)[:max_attribute_nodes]
    )

    projection = nx.Graph()
    for node in selected_attrs:
        projection.add_node(node, **graph.nodes[node])

    edge_counter: Counter[tuple[str, str]] = Counter()
    product_count = 0
    for product, data in graph.nodes(data=True):
        if data.get("node_type") != "PRODUCT":
            continue
        attrs = sorted(neighbor for neighbor in graph.neighbors(product) if neighbor in selected_attrs)
        if len(attrs) < 2:
            continue
        product_count += 1
        for left, right in itertools.combinations(attrs, 2):
            edge_counter[(left, right)] += 1

    for (left, right), weight in edge_counter.items():
        if weight >= min_edge_weight:
            projection.add_edge(left, right, weight=weight)

    projection.graph["product_count"] = product_count
    projection.graph["min_edge_weight"] = min_edge_weight
    return projection


def layered_attribute_layout(graph) -> dict[str, tuple[float, float]]:
    by_type: dict[str, list[str]] = {node_type: [] for node_type in LAYER_ORDER}
    for node, data in graph.nodes(data=True):
        node_type = data.get("node_type")
        if node_type in by_type:
            by_type[node_type].append(node)

    pos: dict[str, tuple[float, float]] = {}
    for x_index, node_type in enumerate(LAYER_ORDER):
        nodes = sorted(by_type[node_type], key=lambda node: graph.degree(node), reverse=True)
        count = len(nodes)
        if count == 0:
            continue
        y_values = list(reversed([index - (count - 1) / 2 for index in range(count)]))
        for node, y_value in zip(nodes, y_values):
            pos[node] = (x_index * 3.0, y_value * 0.55)
    return pos


def draw_attribute_projection(graph, output_path, mpatches, plt, nx) -> str:
    pos = layered_attribute_layout(graph)
    fig, ax = plt.subplots(figsize=(22, 11), dpi=180)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    edge_widths = [
        max(0.25, min(3.0, math.log1p(data.get("weight", 1)) * 0.45))
        for _, _, data in graph.edges(data=True)
    ]
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#6b7280", alpha=0.28, width=edge_widths)

    for node_type in LAYER_ORDER:
        nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == node_type]
        sizes = [260 + graph.degree(node) * 13 for node in nodes]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=NODE_COLORS[node_type],
            node_size=sizes, edgecolors="#f8fafc", linewidths=1.2, alpha=0.96, ax=ax,
        )

    labels = {node: str(data.get("label", node))[:18] for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6, font_color="#f8fafc", font_weight="bold", ax=ax)

    for x_index, node_type in enumerate(LAYER_ORDER):
        ax.text(
            x_index * 3.0, ax.get_ylim()[1] + 0.25, node_type.replace("_", " ").lower(),
            color=NODE_COLORS[node_type], fontsize=10, fontweight="bold", ha="center",
        )

    title = (
        "G_attr - proyeccion estatica de atributos "
        f"({graph.number_of_nodes()} nodos, {graph.number_of_edges()} conexiones, "
        f"{graph.graph.get('product_count', 0)} productos)"
    )
    ax.set_title(title, color="#f8fafc", fontsize=14, pad=18)
    legend = [mpatches.Patch(color=NODE_COLORS[node_type], label=node_type) for node_type in LAYER_ORDER]
    ax.legend(handles=legend, loc="lower right", facecolor="#111111", edgecolor="#4b5563", labelcolor="#f8fafc")
    ax.axis("off")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(output)


def full_attribute_layout(graph) -> dict[str, tuple[float, float]]:
    by_type: dict[str, list[str]] = {"PRODUCT": []}
    by_type.update({node_type: [] for node_type in LAYER_ORDER})
    for node, data in graph.nodes(data=True):
        node_type = data.get("node_type")
        if node_type in by_type:
            by_type[node_type].append(node)

    pos: dict[str, tuple[float, float]] = {}
    columns = [("PRODUCT", -2.2)] + [(node_type, index * 2.4) for index, node_type in enumerate(LAYER_ORDER)]
    for node_type, x in columns:
        nodes = sorted(by_type[node_type], key=lambda node: graph.degree(node), reverse=True)
        count = len(nodes)
        if count == 0:
            continue
        rows_per_lane = 88 if node_type == "PRODUCT" else 44
        lane_count = max(1, math.ceil(count / rows_per_lane))
        for index, node in enumerate(nodes):
            lane = index // rows_per_lane
            row = index % rows_per_lane
            lane_offset = (lane - (lane_count - 1) / 2) * 0.18
            y = ((rows_per_lane - 1) / 2 - row) * 0.16
            pos[node] = (x + lane_offset, y)
    return pos


def draw_full_attribute_graph(graph, output_path, mpatches, plt, nx) -> str:
    pos = full_attribute_layout(graph)
    fig, ax = plt.subplots(figsize=(30, 17), dpi=180)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#64748b", alpha=0.18, width=0.28)

    for node_type, color in NODE_COLORS.items():
        nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]
        if not nodes:
            continue
        if node_type == "PRODUCT":
            sizes = [12 + min(graph.degree(node), 18) * 0.7 for node in nodes]
            edge_color = "#94a3b8"
            alpha = 0.62
        else:
            sizes = [70 + min(graph.degree(node), 80) * 2.6 for node in nodes]
            edge_color = "#f8fafc"
            alpha = 0.94
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=color, node_size=sizes,
            edgecolors=edge_color, linewidths=0.35, alpha=alpha, ax=ax,
        )

    label_candidates = [
        node for node, data in graph.nodes(data=True)
        if data.get("node_type") != "PRODUCT"
    ]
    label_nodes = sorted(label_candidates, key=lambda node: graph.degree(node), reverse=True)[:120]
    labels = {node: str(graph.nodes[node].get("label", node))[:20] for node in label_nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=5.5, font_color="#f8fafc", font_weight="bold", ax=ax)

    for node_type, x in [("PRODUCT", -2.2)] + [(node_type, index * 2.4) for index, node_type in enumerate(LAYER_ORDER)]:
        if any(data.get("node_type") == node_type for _, data in graph.nodes(data=True)):
            ax.text(
                x, ax.get_ylim()[1] + 0.3, node_type.replace("_", " "),
                color=NODE_COLORS[node_type], fontsize=10, fontweight="bold", ha="center",
            )

    ax.set_title(
        f"G_attr - grafo completo ({graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas)",
        color="#f8fafc", fontsize=16, fontweight="bold", loc="left", pad=18,
    )
    legend = [
        mpatches.Patch(color=color, label=node_type)
        for node_type, color in NODE_COLORS.items()
        if any(data.get("node_type") == node_type for _, data in graph.nodes(data=True))
    ]
    ax.legend(handles=legend, loc="lower right", facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc", ncol=2)
    ax.axis("off")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(output)


def draw_focus_product_attribute_graph(graph, output_path, mpatches, plt, nx) -> str:
    pos = nx.spring_layout(graph, seed=42, k=0.85, iterations=180, weight="weight")
    fig, ax = plt.subplots(figsize=(18, 12), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#cbd5e1", alpha=0.45, width=0.8)

    for node_type, color in NODE_COLORS.items():
        nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == node_type]
        if not nodes:
            continue
        if node_type == "PRODUCT":
            sizes = [22 + min(product_activity_score(graph, node), 40) * 0.8 for node in nodes]
            alpha = 0.38
        else:
            sizes = [420 + graph.degree(node) * 18 for node in nodes]
            alpha = 0.96
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=color, node_size=sizes,
            edgecolors="#334155" if node_type != "PRODUCT" else "#94a3b8",
            linewidths=0.8, alpha=alpha, ax=ax,
        )

    labels = {
        node: str(data.get("label", node))[:22]
        for node, data in graph.nodes(data=True)
        if data.get("node_type") != "PRODUCT"
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color="#0f172a", font_weight="bold", ax=ax)

    title = (
        "G_attr - subgrafo estatico de productos y atributos principales "
        f"({graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas)"
    )
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#111827", pad=16)
    legend = [
        mpatches.Patch(color=color, label=node_type)
        for node_type, color in NODE_COLORS.items()
        if any(data.get("node_type") == node_type for _, data in graph.nodes(data=True))
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True)
    ax.axis("off")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(output)


def draw_attribute_family_graph(graph, output_path, mpatches, plt, nx, center_nodes: list[str] | None = None) -> str:
    center_nodes = center_nodes or ["TYPE:FRASCO", "MATERIAL:VIDRIO", "COLOR:AMBAR", "CAPACITY:10ML"]
    selected = set(node for node in center_nodes if node in graph)
    for center in list(selected):
        products = [
            neighbor for neighbor in graph.neighbors(center)
            if graph.nodes[neighbor].get("node_type") == "PRODUCT"
        ]
        products = sorted(products, key=lambda node: product_activity_score(graph, node), reverse=True)[:35]
        selected.update(products)
        for product in products:
            selected.update(
                neighbor for neighbor in graph.neighbors(product)
                if graph.nodes[neighbor].get("node_type") != "PRODUCT"
            )

    subgraph = graph.subgraph(selected).copy()
    return draw_focus_product_attribute_graph(subgraph, output_path, mpatches, plt, nx)


def select_transaction_overview_graph(graph, max_primary: int = 10, max_products: int = 34, max_documents: int = 32):
    """Subgrafo secundario: contraparte(s) centrales + productos/docs conectados."""
    if graph.number_of_nodes() == 0:
        return graph.copy()

    primary_types = {"CLIENT", "SUPPLIER"}
    primary = [
        node for node, data in graph.nodes(data=True)
        if data.get("node_type") in primary_types
    ]
    if not primary:
        primary = [node for node in graph.nodes if not str(node).startswith("PRODUCT:")]
    primary = sorted(primary, key=lambda node: graph.degree(node), reverse=True)[:max_primary]

    selected = set(primary)
    products: set[str] = set()
    documents: set[str] = set()
    for node in primary:
        for neighbor in graph.neighbors(node):
            ntype = graph.nodes[neighbor].get("node_type")
            if ntype == "PRODUCT":
                products.add(neighbor)
            elif ntype == "DOCUMENT":
                documents.add(neighbor)
                products.update(
                    n for n in graph.neighbors(neighbor)
                    if graph.nodes[n].get("node_type") == "PRODUCT"
                )

    products = set(sorted(products, key=lambda node: graph.degree(node), reverse=True)[:max_products])
    documents = set(sorted(documents, key=lambda node: graph.degree(node), reverse=True)[:max_documents])
    selected.update(products)
    selected.update(documents)

    # Si el grafo directo CLIENT/SUPPLIER -> PRODUCT domina, incluir también
    # vecinos secundarios de los productos para que se vea la conectividad.
    if len(selected) < 28:
        for product in list(products)[:16]:
            neighbors = sorted(graph.neighbors(product), key=lambda node: graph.degree(node), reverse=True)
            selected.update(neighbors[:4])
            if len(selected) >= 70:
                break

    return graph.subgraph(selected).copy()


def transaction_layout(graph, nx) -> dict[str, tuple[float, float]]:
    """Layout radial por tipo para lectura consistente en láminas estáticas."""
    by_type: dict[str, list[str]] = {}
    for node, data in graph.nodes(data=True):
        by_type.setdefault(data.get("node_type", "OTHER"), []).append(node)

    if "CLIENT" in by_type or "SUPPLIER" in by_type:
        left_types = ["CLIENT", "SUPPLIER"]
    else:
        left_types = []
    columns = [
        (left_types, -3.2),
        (["DOCUMENT"], 0.0),
        (["PRODUCT"], 3.2),
    ]
    pos: dict[str, tuple[float, float]] = {}
    for types, x in columns:
        nodes: list[str] = []
        for node_type in types:
            nodes.extend(by_type.get(node_type, []))
        nodes = sorted(nodes, key=lambda node: graph.degree(node), reverse=True)
        if not nodes:
            continue
        count = len(nodes)
        y_values = list(reversed([index - (count - 1) / 2 for index in range(count)]))
        spacing = max(0.33, min(0.72, 18 / max(count, 1)))
        for node, y in zip(nodes, y_values):
            pos[node] = (x, y * spacing)

    remaining = [node for node in graph.nodes if node not in pos]
    if remaining:
        spring_pos = nx.spring_layout(graph.subgraph(remaining), seed=42)
        for node, (x, y) in spring_pos.items():
            pos[node] = (float(x), float(y))
    return pos


def transaction_full_layout(graph) -> dict[str, tuple[float, float]]:
    """Layout completo por tipo: conserva todos los nodos sin costo de force layout."""
    by_type: dict[str, list[str]] = {}
    for node, data in graph.nodes(data=True):
        by_type.setdefault(data.get("node_type", "OTHER"), []).append(node)

    x_by_type = {
        "CLIENT": -4.4,
        "SUPPLIER": -3.2,
        "DOCUMENT": 0.0,
        "PRODUCT": 4.0,
        "OTHER": 1.8,
    }
    rows_by_type = {
        "CLIENT": 72,
        "SUPPLIER": 72,
        "DOCUMENT": 96,
        "PRODUCT": 86,
        "OTHER": 80,
    }
    pos: dict[str, tuple[float, float]] = {}
    for node_type in ["CLIENT", "SUPPLIER", "DOCUMENT", "PRODUCT", "OTHER"]:
        nodes = sorted(by_type.get(node_type, []), key=lambda node: graph.degree(node), reverse=True)
        if not nodes:
            continue
        rows_per_lane = rows_by_type[node_type]
        lane_count = max(1, math.ceil(len(nodes) / rows_per_lane))
        for index, node in enumerate(nodes):
            lane = index // rows_per_lane
            row = index % rows_per_lane
            lane_offset = (lane - (lane_count - 1) / 2) * 0.15
            y = ((rows_per_lane - 1) / 2 - row) * 0.14
            pos[node] = (x_by_type[node_type] + lane_offset, y)
    return pos


def draw_transaction_full_graph(graph, output_path, title: str, mpatches, plt, nx) -> str:
    fig, ax = plt.subplots(figsize=(30, 17), dpi=180)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    pos = transaction_full_layout(graph)
    edge_widths = []
    for _, _, data in graph.edges(data=True):
        amount = abs(_to_float(data.get("amount", 0), 0.0))
        weight = abs(_to_float(data.get("weight", 1), 1.0))
        basis = amount if amount > 0 else weight
        edge_widths.append(max(0.08, min(0.85, math.log1p(basis) * 0.09)))

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color="#94a3b8", alpha=0.12,
        width=edge_widths,
    )

    for node_type, color in NODE_COLORS.items():
        nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]
        if not nodes:
            continue
        base_size = 9 if node_type in {"PRODUCT", "DOCUMENT"} else 16
        sizes = [base_size + min(graph.degree(node), 40) * 0.65 for node in nodes]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=color, node_size=sizes,
            edgecolors=color, linewidths=0.1, alpha=0.78, ax=ax,
        )

    label_nodes = sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=True)[:80]
    labels = {
        node: str(graph.nodes[node].get("label", node))[:22]
        for node in label_nodes
    }
    nx.draw_networkx_labels(
        graph, pos, labels=labels, font_size=5.3, font_color="#f8fafc",
        font_weight="bold", ax=ax,
    )

    for node_type, x in {
        "CLIENT": -4.4,
        "SUPPLIER": -3.2,
        "DOCUMENT": 0.0,
        "PRODUCT": 4.0,
        "OTHER": 1.8,
    }.items():
        if any(data.get("node_type") == node_type for _, data in graph.nodes(data=True)):
            ax.text(
                x, ax.get_ylim()[1] + 0.3, node_type,
                color=NODE_COLORS.get(node_type, "#94a3c4"),
                fontsize=11, fontweight="bold", ha="center",
            )

    ax.set_title(
        f"{title} - grafo completo ({graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas)",
        color="#f8fafc", fontsize=16, fontweight="bold", loc="left", pad=18,
    )
    legend = [
        mpatches.Patch(color=color, label=node_type)
        for node_type, color in NODE_COLORS.items()
        if any(data.get("node_type") == node_type for _, data in graph.nodes(data=True))
    ]
    if legend:
        ax.legend(handles=legend, loc="lower right", facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc")
    ax.axis("off")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(output)


def draw_transaction_overview_graph(graph, output_path, title: str, mpatches, plt, nx) -> str:
    fig, ax = plt.subplots(figsize=(22, 13), dpi=180)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    pos = transaction_layout(graph, nx)
    edge_widths = []
    for _, _, data in graph.edges(data=True):
        amount = abs(_to_float(data.get("amount", 0), 0.0))
        weight = abs(_to_float(data.get("weight", 1), 1.0))
        basis = amount if amount > 0 else weight
        edge_widths.append(max(0.35, min(3.2, math.log1p(basis) * 0.32)))

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color="#64748b", alpha=0.34,
        width=edge_widths,
    )

    for node_type, color in NODE_COLORS.items():
        nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]
        if not nodes:
            continue
        sizes = [260 + min(graph.degree(node), 28) * 22 for node in nodes]
        if node_type == "PRODUCT":
            sizes = [190 + min(graph.degree(node), 28) * 16 for node in nodes]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=color, node_size=sizes,
            edgecolors="#dbeafe", linewidths=1.0, alpha=0.94, ax=ax,
        )

    label_nodes = sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=True)[:42]
    labels = {
        node: str(graph.nodes[node].get("label", node))[:24]
        for node in label_nodes
    }
    nx.draw_networkx_labels(
        graph, pos, labels=labels, font_size=7, font_color="#f8fafc",
        font_weight="bold", ax=ax,
    )

    ax.set_title(
        f"{title} - muestra estatica ({graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas)",
        color="#f8fafc", fontsize=15, fontweight="bold", loc="left", pad=18,
    )
    legend = [
        mpatches.Patch(color=color, label=node_type)
        for node_type, color in NODE_COLORS.items()
        if any(data.get("node_type") == node_type for _, data in graph.nodes(data=True))
    ]
    if legend:
        ax.legend(handles=legend, loc="lower right", facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc")
    ax.axis("off")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(output)


def render_graph_visualizations(stage2_output_dir: str | Path, output_dir: str | Path) -> dict[str, str]:
    mpatches, plt, nx = _import_libs()
    nodes, edges, metrics = load_graph_tables(stage2_output_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if not nodes.empty and not edges.empty:
        graph = build_networkx_graph(nodes, edges, nx)
        focus_graph = select_focus_product_attribute_graph(graph)
        attribute_projection = build_attribute_projection(graph)
        paths.update(
            {
                "g_attr_full.png": draw_full_attribute_graph(
                    graph, output / "g_attr_full.png", mpatches, plt, nx
                ),
                "g_attr_attribute_projection.png": draw_attribute_projection(
                    attribute_projection, output / "g_attr_attribute_projection.png", mpatches, plt, nx
                ),
                "g_attr_product_attribute_focus.png": draw_focus_product_attribute_graph(
                    focus_graph, output / "g_attr_product_attribute_focus.png", mpatches, plt, nx
                ),
                "g_attr_frasco_vidrio_ambar.png": draw_attribute_family_graph(
                    graph, output / "g_attr_frasco_vidrio_ambar.png", mpatches, plt, nx
                ),
            }
        )

    transaction_titles = {
        "sales": "G_sales - ventas",
        "purchases": "G_purchases - compras",
        "business": "G_business - negocio combinado",
    }
    for name, title in transaction_titles.items():
        tx_nodes, tx_edges, _tx_metrics = load_named_graph_tables(stage2_output_dir, name)
        if tx_nodes.empty or tx_edges.empty:
            continue
        tx_graph = build_networkx_graph(tx_nodes, tx_edges, nx)
        full_filename = f"g_{name}_full.png"
        paths[full_filename] = draw_transaction_full_graph(
            tx_graph, output / full_filename, title, mpatches, plt, nx
        )
        overview = select_transaction_overview_graph(tx_graph)
        filename = f"g_{name}_overview.png"
        paths[filename] = draw_transaction_overview_graph(
            overview, output / filename, title, mpatches, plt, nx
        )

    manifest_path = output / "visualization_manifest.json"
    manifest = {
        "graph_name": metrics.get("graph_name", "G_attr"),
        "layout": "static_networkx_png",
        "outputs": {name: str(path) for name, path in paths.items()},
        "notes": [
            "No HTML interactivo; instantáneas para presentación/documentación.",
            "Los archivos *_full.png visualizan el grafo completo generado desde los CSV del dataset.",
            "Los archivos *_overview.png y subgrafos de G_attr son vistas secundarias para lectura rápida.",
            "Los valores exactos de nodos, aristas y pesos están en los CSV.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["visualization_manifest.json"] = str(manifest_path)
    return paths
