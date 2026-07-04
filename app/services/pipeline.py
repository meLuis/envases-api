from __future__ import annotations

from pathlib import Path
import shutil
from typing import BinaryIO

import pandas as pd

from app.config import (
    ENABLE_GRAPH_IMAGES,
    MIN_ATTRIBUTE_COVERAGE,
    MIN_GRAPH_EDGES,
    MIN_PRODUCT_ROWS,
    MIN_TRANSACTION_ROWS,
)
from app.core.algorithms import bellman_ford_savings, build_supply_options, families_from_projection
from app.core.attributes import extract_product_attributes
from app.core.attribute_llm import gemini_available, refine_rules_with_gemini
from app.core.graphs import build_product_projection, build_semantic_graph, build_transaction_graphs
from app.core.io import read_table
from app.core.normalization import normalize_all, profile_frames, quality_summary
from app.domain.schemas import ArtifactStatus, DatasetSummary
from app.storage.repository import dataset_dir, ensure_dataset_dir, new_dataset_id, write_csv, write_json


class PipelineError(ValueError):
    pass


def build_dataset_from_uploads(
    products_file: BinaryIO,
    sales_file: BinaryIO,
    purchases_file: BinaryIO,
    product_filename: str,
    sales_filename: str,
    purchases_filename: str,
) -> DatasetSummary:
    dataset_id = new_dataset_id()
    dataset_path = ensure_dataset_dir(dataset_id)
    (dataset_path / "raw").mkdir(exist_ok=True)

    raw = {
        "products": read_table(products_file, product_filename),
        "sales": read_table(sales_file, sales_filename),
        "purchases": read_table(purchases_file, purchases_filename),
    }
    _save_upload(products_file, dataset_path / "raw" / product_filename)
    _save_upload(sales_file, dataset_path / "raw" / sales_filename)
    _save_upload(purchases_file, dataset_path / "raw" / purchases_filename)
    return _build_dataset(dataset_id, raw)


def build_dataset_from_paths(products_path: Path, sales_path: Path, purchases_path: Path) -> DatasetSummary:
    dataset_id = new_dataset_id()
    raw = {
        "products": read_table(products_path),
        "sales": read_table(sales_path),
        "purchases": read_table(purchases_path),
    }
    return _build_dataset(dataset_id, raw)


def _build_dataset(dataset_id: str, raw: dict[str, pd.DataFrame]) -> DatasetSummary:
    generated: list[ArtifactStatus] = []
    omitted: list[ArtifactStatus] = []
    warnings: list[str] = []

    _validate_minimum(raw)
    cleaned, mapping = normalize_all(raw)
    profiles = profile_frames(raw)

    for key, filename in {
        "products": "products_clean.csv",
        "sales": "sales_clean.csv",
        "purchases": "purchases_clean.csv",
    }.items():
        write_csv(dataset_id, filename, cleaned[key])
        generated.append(ArtifactStatus(name=filename, kind="base"))

    write_json(dataset_id, "schema_mapping.json", mapping)
    write_json(dataset_id, "profiles.json", profiles)
    write_json(dataset_id, "normalization_report.json", _normalization_report(cleaned, mapping))
    write_json(dataset_id, "company_rules.json", _company_rules())
    for name in ["schema_mapping.json", "profiles.json", "normalization_report.json", "company_rules.json"]:
        generated.append(ArtifactStatus(name=name, kind="base"))

    write_csv(dataset_id, "quality_summary.csv", quality_summary(cleaned))
    write_csv(dataset_id, "product_matches.csv", _product_matches(cleaned["products"], cleaned["sales"], cleaned["purchases"]))
    product_activity = _product_activity(cleaned["sales"], cleaned["purchases"])
    write_csv(dataset_id, "product_activity_summary.csv", product_activity)
    write_csv(dataset_id, "transaction_flags_summary.csv", _transaction_flags(cleaned["sales"], cleaned["purchases"]))
    for name in ["quality_summary.csv", "product_matches.csv", "product_activity_summary.csv", "transaction_flags_summary.csv"]:
        generated.append(ArtifactStatus(name=name, kind="base"))

    attributes, attr_report = extract_product_attributes(cleaned["products"])
    # Refinamiento opcional con Gemini (solo si hay GEMINI_API_KEY): propone
    # reglas nuevas y se aceptan solo si mejoran sin regresión. Failsafe.
    if gemini_available():
        attributes, attr_report, llm_info = refine_rules_with_gemini(cleaned["products"], attributes, attr_report)
        write_json(dataset_id, "attribute_llm_refinement.json", llm_info)
        generated.append(ArtifactStatus(name="attribute_llm_refinement.json", kind="semantic"))
        warnings.append(f"Gemini: {llm_info.get('status')} ({llm_info.get('reason', llm_info.get('source_model', ''))}).")
    if attr_report["coverage"] >= MIN_ATTRIBUTE_COVERAGE:
        write_csv(dataset_id, "product_attributes.csv", attributes)
        write_json(dataset_id, "attribute_extraction_report.json", attr_report)
        write_json(dataset_id, "attribute_rules.json", attr_report["rules"])
        write_csv(dataset_id, "attribute_coverage_report.csv", pd.DataFrame([attr_report]))
        generated.extend(
            ArtifactStatus(name=name, kind="semantic")
            for name in [
                "product_attributes.csv",
                "attribute_extraction_report.json",
                "attribute_rules.json",
                "attribute_coverage_report.csv",
            ]
        )
        semantic_nodes, semantic_edges, semantic_metrics = build_semantic_graph(attributes, activity=product_activity)
        projection_edges, projection_metrics = build_product_projection(attributes)
        if len(semantic_edges) >= MIN_GRAPH_EDGES:
            write_csv(dataset_id, "semantic_attribute_graph_nodes.csv", semantic_nodes)
            write_csv(dataset_id, "semantic_attribute_graph_edges.csv", semantic_edges)
            write_json(dataset_id, "semantic_attribute_graph_metrics.json", semantic_metrics)
            generated.extend(
                ArtifactStatus(name=name, kind="graph")
                for name in [
                    "semantic_attribute_graph_nodes.csv",
                    "semantic_attribute_graph_edges.csv",
                    "semantic_attribute_graph_metrics.json",
                ]
            )
        else:
            omitted.append(ArtifactStatus(name="G_attr", kind="graph", generated=False, reason="Menos aristas que el minimo configurado."))
        if len(projection_edges) >= MIN_GRAPH_EDGES:
            write_csv(dataset_id, "product_projection_edges.csv", projection_edges)
            write_json(dataset_id, "product_projection_metrics.json", projection_metrics)
            top = projection_edges.head(100) if not projection_edges.empty else projection_edges
            write_csv(dataset_id, "product_projection_top_similar.csv", top)
            families = families_from_projection(projection_edges)
            write_csv(dataset_id, "ufds_product_families.csv", families)
            generated.extend(
                ArtifactStatus(name=name, kind="graph")
                for name in [
                    "product_projection_edges.csv",
                    "product_projection_metrics.json",
                    "product_projection_top_similar.csv",
                    "ufds_product_families.csv",
                ]
            )
        else:
            omitted.append(ArtifactStatus(name="G_projection", kind="graph", generated=False, reason="La similitud entre productos no genero suficientes aristas."))
    else:
        omitted.append(ArtifactStatus(name="semantic_outputs", kind="semantic", generated=False, reason="La cobertura de atributos no alcanzo el minimo."))

    if len(cleaned["sales"]) >= MIN_TRANSACTION_ROWS and len(cleaned["purchases"]) >= MIN_TRANSACTION_ROWS:
        graphs = build_transaction_graphs(cleaned["sales"], cleaned["purchases"])
        for name, (nodes, edges, metrics) in graphs.items():
            write_csv(dataset_id, f"transaction_graph_{name}_nodes.csv", nodes)
            write_csv(dataset_id, f"transaction_graph_{name}_edges.csv", edges)
            write_json(dataset_id, f"transaction_graph_{name}_metrics.json", metrics)
            generated.extend(
                ArtifactStatus(name=artifact, kind="graph")
                for artifact in [
                    f"transaction_graph_{name}_nodes.csv",
                    f"transaction_graph_{name}_edges.csv",
                    f"transaction_graph_{name}_metrics.json",
                ]
            )
    else:
        omitted.append(ArtifactStatus(name="transaction_graphs", kind="graph", generated=False, reason="Ventas o compras no alcanzan filas minimas."))

    # PNG estáticos de grafos principales (opcional, failsafe: nunca bloquea el dataset).
    if ENABLE_GRAPH_IMAGES:
        try:
            from app.core.graph_visualizer import render_graph_visualizations

            dataset_path = dataset_dir(dataset_id)
            images = render_graph_visualizations(dataset_path, dataset_path)
            generated.extend(
                ArtifactStatus(name=name, kind="visualization")
                for name in images
                if (dataset_path / name).exists()
            )
        except Exception as exc:  # noqa: BLE001 — visualización es accesoria
            warnings.append(f"Imágenes de grafos no generadas: {exc}")

    options = build_supply_options(cleaned["purchases"])
    if not options.empty:
        write_csv(dataset_id, "supply_options.csv", options)
        generated.append(ArtifactStatus(name="supply_options.csv", kind="optimization"))
        offers = bellman_ford_savings(options)
        write_csv(dataset_id, "bellman_ford_candidates.csv", offers["candidates"])
        write_csv(dataset_id, "bellman_ford_edges.csv", offers["edges"])
        write_csv(dataset_id, "bellman_ford_best_paths.csv", offers["best_paths"])
        write_json(dataset_id, "bellman_ford_summary.json", offers["summary"])
        generated.extend(
            ArtifactStatus(name=name, kind="algorithm")
            for name in [
                "bellman_ford_candidates.csv",
                "bellman_ford_edges.csv",
                "bellman_ford_best_paths.csv",
                "bellman_ford_summary.json",
            ]
        )
    else:
        omitted.append(ArtifactStatus(name="supply_options", kind="optimization", generated=False, reason="No hay compras validas con costo unitario."))

    summary = DatasetSummary(
        dataset_id=dataset_id,
        status="ready",
        row_counts={key: int(len(value)) for key, value in cleaned.items()},
        generated=generated,
        omitted=omitted,
        warnings=warnings,
    )
    write_json(dataset_id, "dataset_summary.json", summary.model_dump())
    return summary


def _validate_minimum(raw: dict[str, pd.DataFrame]) -> None:
    if len(raw["products"]) < MIN_PRODUCT_ROWS:
        raise PipelineError("Productos no alcanza el minimo de filas para una demo util.")
    for kind in ("sales", "purchases"):
        if len(raw[kind]) < MIN_TRANSACTION_ROWS:
            raise PipelineError(f"{kind} no alcanza el minimo de transacciones.")


def _save_upload(file: BinaryIO, path: Path) -> None:
    if hasattr(file, "seek"):
        file.seek(0)
    with path.open("wb") as output:
        shutil.copyfileobj(file, output)
    if hasattr(file, "seek"):
        file.seek(0)


def _normalization_report(cleaned: dict[str, pd.DataFrame], mapping: dict) -> dict:
    return {
        "row_counts": {key: int(len(value)) for key, value in cleaned.items()},
        "mapped_fields": {
            kind: sum(1 for field in fields.values() if field.get("column"))
            for kind, fields in mapping.items()
        },
    }


def _company_rules() -> dict:
    return {
        "domain": "envases_vidrio_plastico",
        "required_files": ["productos", "ventas", "compras"],
        "llm": "enabled_gemini" if gemini_available() else "disabled",
        "main_algorithms": ["BFS", "BFS bidireccional", "Programacion dinamica", "Bellman-Ford", "UFDS"],
        "extra_algorithm": "Min-cost flow",
    }


def _product_matches(products: pd.DataFrame, sales: pd.DataFrame, purchases: pd.DataFrame) -> pd.DataFrame:
    known = set(products["product_id"].astype(str))
    observed = pd.concat([sales[["product_id", "product_name"]], purchases[["product_id", "product_name"]]], ignore_index=True)
    observed = observed.drop_duplicates("product_id")
    observed["in_catalog"] = observed["product_id"].astype(str).isin(known)
    observed["status"] = observed["in_catalog"].map(lambda ok: "accepted" if ok else "unmatched")
    return observed


def _product_activity(sales: pd.DataFrame, purchases: pd.DataFrame) -> pd.DataFrame:
    sold = sales.groupby("product_id", as_index=False).agg(sold_units=("quantity", "sum"), sales_amount=("total", "sum"))
    bought = purchases.groupby("product_id", as_index=False).agg(purchased_units=("quantity", "sum"), purchase_amount=("total", "sum"))
    return sold.merge(bought, on="product_id", how="outer").fillna(0).sort_values(["sold_units", "purchased_units"], ascending=False)


def _transaction_flags(sales: pd.DataFrame, purchases: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset": "sales", "rows": len(sales), "zero_quantity": int((sales["quantity"] <= 0).sum()), "zero_total": int((sales["total"] <= 0).sum())},
            {"dataset": "purchases", "rows": len(purchases), "zero_quantity": int((purchases["quantity"] <= 0).sum()), "zero_total": int((purchases["total"] <= 0).sum())},
        ]
    )
