"""Imágenes PNG estáticas de G_attr (port del visualizador base, adaptado).

Tres estilos didácticos:
1. Proyección en columnas por capa (TYPE, SUBTYPE, … MOUTH_SIZE).
2. Subgrafo force-directed producto ↔ atributos principales.
3. Familia de atributos (FRASCO + VIDRIO + AMBAR…).

Adaptado al esquema de aristas del backend actual (columna `edge_type` en vez de
`relation`; los nodos no traen actividad, así que la selección cae al grado).
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


def load_graph_tables(stage2_output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = Path(stage2_output_dir)
    nodes = pd.read_csv(base / "semantic_attribute_graph_nodes.csv", encoding="utf-8-sig")
    edges = pd.read_csv(base / "semantic_attribute_graph_edges.csv", encoding="utf-8-sig")
    metrics_path = base / "semantic_attribute_graph_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
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
        try:
            weight = float(edge.get("weight", 1) or 1)
        except (TypeError, ValueError):
            weight = 1.0
        graph.add_edge(edge["source"], edge["target"], weight=weight)
    return graph


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


def render_graph_visualizations(stage2_output_dir: str | Path, output_dir: str | Path) -> dict[str, str]:
    mpatches, plt, nx = _import_libs()
    nodes, edges, metrics = load_graph_tables(stage2_output_dir)
    graph = build_networkx_graph(nodes, edges, nx)
    focus_graph = select_focus_product_attribute_graph(graph)
    attribute_projection = build_attribute_projection(graph)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
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

    manifest_path = output / "visualization_manifest.json"
    manifest = {
        "graph_name": metrics.get("graph_name", "G_attr"),
        "layout": "static_networkx_png",
        "outputs": {name: str(path) for name, path in paths.items()},
        "notes": [
            "No HTML interactivo; instantáneas para presentación/documentación.",
            "La proyección de atributos conecta atributos que aparecen juntos en productos.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["visualization_manifest.json"] = str(manifest_path)
    return paths
