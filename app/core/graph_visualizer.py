"""Imagenes PNG estaticas para los grafos expuestos en el sandbox.

Se genera una lamina principal por estructura visible:
G_attr, G_sales, G_purchases, G_business, G_supplier_projection, G_offers y flow.

Adaptado al esquema de aristas del backend actual (columna `edge_type` en vez de
`relation`; el peso de arista usa `weight`/`amount` cuando existen).
Se importa de forma perezosa para no exigir matplotlib/networkx salvo cuando se
generan imagenes.
"""

from __future__ import annotations

import json
import math
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
    # Por capas: una columna por dimension del producto (TYPE, SUBTYPE, ...) mas
    # una columna de PRODUCT, con carriles (lanes) para no perder nodos cuando
    # una capa tiene cientos de miembros. Incluye TODOS los nodos del grafo.
    pos = full_attribute_layout(graph)
    fig, ax = plt.subplots(figsize=(28, 16), dpi=160)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#64748b", alpha=0.12, width=0.22)

    for node_type, color in NODE_COLORS.items():
        nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]
        if not nodes:
            continue
        if node_type == "PRODUCT":
            sizes = [10 + min(graph.degree(node), 18) * 0.6 for node in nodes]
            edge_color = "#94a3b8"
            alpha = 0.55
        else:
            sizes = [55 + min(graph.degree(node), 80) * 2.0 for node in nodes]
            edge_color = "#f8fafc"
            alpha = 0.94
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=color, node_size=sizes,
            edgecolors=edge_color, linewidths=0.3, alpha=alpha, ax=ax,
        )

    label_candidates = [
        node for node, data in graph.nodes(data=True)
        if data.get("node_type") != "PRODUCT"
    ]
    label_nodes = sorted(label_candidates, key=lambda node: graph.degree(node), reverse=True)[:140]
    labels = {node: str(graph.nodes[node].get("label", node))[:18] for node in label_nodes}

    # Un par de productos de muestra con etiqueta, solo para que se entienda
    # que esa columna gris son productos (el resto queda sin texto por volumen).
    product_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "PRODUCT"]
    sample_products = sorted(product_nodes, key=lambda node: graph.degree(node), reverse=True)[:2]
    for node in sample_products:
        labels[node] = str(graph.nodes[node].get("label", node))[:18]

    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=5.5, font_color="#f8fafc", font_weight="bold", ax=ax)

    column_order = ["PRODUCT"] + LAYER_ORDER
    column_x = {"PRODUCT": -2.2}
    column_x.update({node_type: index * 2.4 for index, node_type in enumerate(LAYER_ORDER)})
    top_y = max((y for _, y in pos.values()), default=0.0) + 1.0
    for node_type in column_order:
        if not any(data.get("node_type") == node_type for _, data in graph.nodes(data=True)):
            continue
        ax.text(
            column_x[node_type], top_y, node_type.replace("_", " ").lower(),
            color=NODE_COLORS[node_type], fontsize=11, fontweight="bold", ha="center",
        )

    ax.set_title(
        f"G_attr - grafo completo por capas ({graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas)",
        color="#f8fafc", fontsize=16, fontweight="bold", loc="left", pad=24,
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




def concentric_layout(rings: list[list[str]]) -> dict[str, tuple[float, float]]:
    """Coloca cada anillo (lista de nodos, de adentro hacia afuera) en un cÃ­rculo.

    Los anillos se espacian de forma UNIFORME (radio = Ã­ndice del anillo): asÃ­
    quedan aros concÃ©ntricos limpios y parejos que llenan el disco, sin huecos
    enormes. El llamador ordena los anillos de menor a mayor cantidad, de modo
    que el aro mÃ¡s interno (menos circunferencia) lleva el grupo mÃ¡s pequeÃ±o.
    """
    pos: dict[str, tuple[float, float]] = {}
    non_empty = [ring for ring in rings if ring]
    radius_step = 1.0
    for ring_index, nodes in enumerate(non_empty):
        radius = (ring_index + 1) * radius_step
        count = len(nodes)
        for index, node in enumerate(nodes):
            angle = 2 * math.pi * index / count - math.pi / 2
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
    return pos


def _rings_by_count(graph, groups: list[list[str]]) -> list[list[str]]:
    """Construye anillos por grupos de tipos, ordenados de menor a mayor tamaÃ±o.

    Cada grupo es una lista de node_types que comparten anillo. Dentro del anillo
    los nodos quedan contiguos por tipo y ordenados por grado (los hubs juntos).
    """
    by_type: dict[str, list[str]] = {}
    for node, data in graph.nodes(data=True):
        by_type.setdefault(data.get("node_type", "OTHER"), []).append(node)

    placed: set[str] = set()
    rings: list[list[str]] = []
    for group in groups:
        ring: list[str] = []
        for node_type in group:
            members = sorted(by_type.get(node_type, []), key=lambda n: graph.degree(n), reverse=True)
            ring.extend(members)
            placed.update(members)
        if ring:
            rings.append(ring)
    # Cualquier tipo no contemplado va en un anillo extra.
    leftover = [n for n in graph.nodes if n not in placed]
    if leftover:
        rings.append(sorted(leftover, key=lambda n: graph.degree(n), reverse=True))
    rings.sort(key=len)  # menor adentro, mayor afuera
    return rings


def draw_transaction_full_graph(graph, output_path, title: str, mpatches, plt, nx) -> str:
    fig, ax = plt.subplots(figsize=(24, 24), dpi=180)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    # Layout circular: anillos concÃ©ntricos por tipo (entidades dentro, el grupo
    # mÃ¡s numeroso en el aro exterior).
    pos = concentric_layout(
        _rings_by_count(graph, [["CLIENT"], ["SUPPLIER"], ["DOCUMENT"], ["PRODUCT"]])
    )
    edge_widths = []
    for _, _, data in graph.edges(data=True):
        amount = abs(_to_float(data.get("amount", 0), 0.0))
        weight = abs(_to_float(data.get("weight", 1), 1.0))
        basis = amount if amount > 0 else weight
        edge_widths.append(max(0.08, min(0.85, math.log1p(basis) * 0.09)))

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color="#94a3b8", alpha=0.05,
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

    ax.set_aspect("equal")
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


def _draw_generic_static_graph(graph, output_path, title: str, mpatches, plt, nx) -> str:
    fig, ax = plt.subplots(figsize=(24, 16), dpi=170)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    if graph.number_of_nodes() == 0:
        ax.set_title(f"{title} - sin nodos", color="#f8fafc", fontsize=16, fontweight="bold", loc="left")
        ax.axis("off")
    else:
        pos = nx.spring_layout(graph, seed=42, k=1.3, iterations=180, weight="layout_weight")
        widths = [
            max(0.25, min(3.2, math.log1p(abs(_to_float(data.get("weight", 1), 1.0))) * 0.45))
            for _, _, data in graph.edges(data=True)
        ]
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#64748b", alpha=0.2, width=widths)

        drawn: set[str] = set()
        for node_type, color in NODE_COLORS.items():
            nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]
            if not nodes:
                continue
            drawn.update(nodes)
            sizes = [160 + min(graph.degree(node), 40) * 18 for node in nodes]
            nx.draw_networkx_nodes(
                graph, pos, nodelist=nodes, node_color=color, node_size=sizes,
                edgecolors="#dbeafe", linewidths=0.8, alpha=0.94, ax=ax,
            )

        other_nodes = [node for node in graph.nodes if node not in drawn]
        if other_nodes:
            sizes = [140 + min(graph.degree(node), 40) * 14 for node in other_nodes]
            nx.draw_networkx_nodes(
                graph, pos, nodelist=other_nodes, node_color="#94a3b8", node_size=sizes,
                edgecolors="#dbeafe", linewidths=0.8, alpha=0.9, ax=ax,
            )

        label_nodes = sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=True)[:80]
        labels = {node: str(graph.nodes[node].get("label", node))[:24] for node in label_nodes}
        nx.draw_networkx_labels(
            graph, pos, labels=labels, font_size=6.2, font_color="#f8fafc",
            font_weight="bold", ax=ax,
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


def build_supplier_projection_graph(edges: pd.DataFrame, nx) -> Any:
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)
        source_name = str(getattr(row, "source_name", source) or source)
        target_name = str(getattr(row, "target_name", target) or target)
        similarity = _to_float(getattr(row, "similarity", 0), 0.0)
        graph.add_node(source, label=source_name, node_type="SUPPLIER")
        graph.add_node(target, label=target_name, node_type="SUPPLIER")
        graph.add_edge(
            source,
            target,
            weight=similarity,
            layout_weight=max(similarity, 0.01),
            edge_type="supplier_similarity",
        )
    return graph


def build_offers_graph(edges: pd.DataFrame, nx) -> Any:
    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)
        weight = _to_float(getattr(row, "weight", 0), 0.0)
        supplier = str(getattr(row, "supplier", "") or "")
        source_type = "OTHER" if source == "SOURCE" else "PRODUCT"
        target_type = "SUPPLIER" if target.startswith("OPTION:") else "PRODUCT"
        graph.add_node(source, label=source.replace("PRODUCT:", ""), node_type=source_type)
        graph.add_node(target, label=supplier or target.replace("OPTION:", ""), node_type=target_type)
        graph.add_edge(
            source,
            target,
            weight=weight,
            layout_weight=1.0 / (abs(weight) + 1.0),
            edge_type=str(getattr(row, "edge_type", "")),
        )
    return graph


def build_flow_graph(options: pd.DataFrame, nx, max_products: int = 50, max_suppliers: int = 20) -> Any:
    graph = nx.DiGraph()
    graph.add_node("SOURCE", label="SOURCE", node_type="OTHER")
    graph.add_node("SINK", label="SINK", node_type="OTHER")
    if options.empty:
        return graph
    options = options.copy()
    for column in ("capacity_units", "unit_cost"):
        if column in options:
            options[column] = pd.to_numeric(options[column], errors="coerce").fillna(0)

    ranked_products = (
        options.groupby(["product_id", "product_name"], as_index=False)
        .agg(total_capacity=("capacity_units", "sum"))
        .sort_values("total_capacity", ascending=False)
        .head(max_products)
    )
    selected_products = set(ranked_products["product_id"].astype(str))
    supplier_capacity = (
        options.groupby("supplier", as_index=False)
        .agg(total_capacity=("capacity_units", "sum"))
        .sort_values("total_capacity", ascending=False)
        .head(max_suppliers)
    )
    selected_suppliers = set(supplier_capacity["supplier"].astype(str))

    for row in ranked_products.itertuples(index=False):
        product = f"SKU:{row.product_id}"
        graph.add_node(product, label=str(row.product_name)[:30], node_type="PRODUCT")
        graph.add_edge("SOURCE", product, weight=float(row.total_capacity or 0), layout_weight=1.0, edge_type="demand")

    filtered = options[
        options["product_id"].astype(str).isin(selected_products)
        & options["supplier"].astype(str).isin(selected_suppliers)
    ]
    for row in filtered.itertuples(index=False):
        product = f"SKU:{row.product_id}"
        supplier = f"SUPPLIER:{row.supplier}"
        graph.add_node(supplier, label=str(row.supplier)[:30], node_type="SUPPLIER")
        unit_cost = _to_float(getattr(row, "unit_cost", 0), 0.0)
        graph.add_edge(
            product,
            supplier,
            weight=unit_cost,
            layout_weight=1.0 / (unit_cost + 1.0),
            edge_type="supplier_option",
        )
    for supplier in selected_suppliers:
        supplier_node = f"SUPPLIER:{supplier}"
        if supplier_node in graph:
            cap = supplier_capacity.loc[supplier_capacity["supplier"].astype(str) == supplier, "total_capacity"].iloc[0]
            graph.add_edge(supplier_node, "SINK", weight=float(cap or 0), layout_weight=1.0, edge_type="capacity")
    return graph


def render_graph_visualizations(stage2_output_dir: str | Path, output_dir: str | Path) -> dict[str, str]:
    mpatches, plt, nx = _import_libs()
    nodes, edges, metrics = load_graph_tables(stage2_output_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if not nodes.empty and not edges.empty:
        graph = build_networkx_graph(nodes, edges, nx)
        paths.update(
            {
                "g_attr_full.png": draw_full_attribute_graph(
                    graph, output / "g_attr_full.png", mpatches, plt, nx
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

    supplier_edges = _safe_read_csv(Path(stage2_output_dir) / "supplier_projection_edges.csv")
    if not supplier_edges.empty:
        supplier_graph = build_supplier_projection_graph(supplier_edges, nx)
        paths["g_supplier_projection_full.png"] = _draw_generic_static_graph(
            supplier_graph,
            output / "g_supplier_projection_full.png",
            "G_supplier_projection - proveedores similares",
            mpatches,
            plt,
            nx,
        )

    offer_edges = _safe_read_csv(Path(stage2_output_dir) / "bellman_ford_edges.csv")
    if not offer_edges.empty:
        offers_graph = build_offers_graph(offer_edges, nx)
        paths["g_offers_full.png"] = _draw_generic_static_graph(
            offers_graph,
            output / "g_offers_full.png",
            "G_offers - ahorros Bellman-Ford",
            mpatches,
            plt,
            nx,
        )

    supply_options = _safe_read_csv(Path(stage2_output_dir) / "supply_options.csv")
    if not supply_options.empty:
        flow_graph = build_flow_graph(supply_options, nx)
        paths["g_flow_full.png"] = _draw_generic_static_graph(
            flow_graph,
            output / "g_flow_full.png",
            "flow - red de optimizacion de compras",
            mpatches,
            plt,
            nx,
        )

    manifest_path = output / "visualization_manifest.json"
    manifest = {
        "graph_name": metrics.get("graph_name", "G_attr"),
        "layout": "static_networkx_png",
        "outputs": {name: str(path) for name, path in paths.items()},
        "notes": [
            "No HTML interactivo; instantÃ¡neas para presentaciÃ³n/documentaciÃ³n.",
            "Los archivos *_full.png corresponden a las estructuras disponibles en el sandbox algoritmo x grafo.",
            "Los valores exactos de nodos, aristas y pesos estÃ¡n en los CSV.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["visualization_manifest.json"] = str(manifest_path)
    return paths
