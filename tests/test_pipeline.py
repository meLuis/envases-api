from __future__ import annotations

from pathlib import Path

from app.services.pipeline import PipelineError, build_dataset_from_paths


BASE = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_build_complete_fixture() -> None:
    summary = build_dataset_from_paths(
        BASE / "completo" / "productos.csv",
        BASE / "completo" / "ventas.csv",
        BASE / "completo" / "compras.csv",
    )
    assert summary.status == "ready"
    assert summary.row_counts["products"] > 0
    assert any(item.name == "transaction_graph_business_edges.csv" for item in summary.generated)


def test_incomplete_fixture_rejected() -> None:
    try:
        build_dataset_from_paths(
            BASE / "incompleto" / "productos.csv",
            BASE / "incompleto" / "ventas.csv",
            BASE / "incompleto" / "compras.csv",
        )
    except PipelineError:
        return
    raise AssertionError("El fixture incompleto debia ser rechazado.")

