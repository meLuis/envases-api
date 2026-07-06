from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import ENABLE_GRAPH_IMAGES
from app.domain.schemas import BudgetRequest, MinCostFlowRequest, QueryResponse
from app.services import queries
from app.services.pipeline import PipelineError, build_dataset_from_uploads
from app.storage.repository import list_artifact_files, require_dataset_dir, list_all_datasets, delete_dataset


router = APIRouter()

GRAPH_VISUALIZATION_ARTIFACTS = {
    "g_attr_full.png",
    "g_sales_full.png",
    "g_purchases_full.png",
    "g_business_full.png",
    "g_supplier_projection_full.png",
    "g_offers_full.png",
    "g_flow_full.png",
    "visualization_manifest.json",
}


@router.get("/health")
def health() -> dict:
    return {"ok": True, "service": "envases_backend"}


@router.get("/datasets")
def get_all_datasets() -> dict:
    datasets = list_all_datasets()
    return {"datasets": datasets, "count": len(datasets)}


@router.post("/datasets")
async def create_dataset(
    productos: UploadFile = File(...),
    ventas: UploadFile = File(...),
    compras: UploadFile = File(...),
):
    try:
        return build_dataset_from_uploads(
            productos.file,
            ventas.file,
            compras.file,
            productos.filename or "productos.csv",
            ventas.filename or "ventas.csv",
            compras.filename or "compras.csv",
        )
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo construir el dataset: {exc}") from exc


@router.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: str) -> dict:
    try:
        delete_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": dataset_id}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        return queries.dataset_summary(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/artifacts")
def list_artifacts(dataset_id: str) -> dict:
    try:
        files = list_artifact_files(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "dataset_id": dataset_id,
        "artifacts": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "download_url": f"/datasets/{dataset_id}/artifacts/{path.name}",
            }
            for path in files
        ],
    }


@router.get("/datasets/{dataset_id}/artifacts/{artifact_name}")
def get_artifact(dataset_id: str, artifact_name: str):
    try:
        path = require_dataset_dir(dataset_id) / artifact_name
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if ENABLE_GRAPH_IMAGES and artifact_name in GRAPH_VISUALIZATION_ARTIFACTS and not path.exists():
        try:
            from app.core.graph_visualizer import render_graph_visualizations

            render_graph_visualizations(path.parent, path.parent)
        except Exception:  # noqa: BLE001 - la descarga debe responder 404 si la visualización no se pudo crear
            pass
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artefacto no encontrado.")
    # Los artefactos son INMUTABLES: el dataset_id es una dirección de contenido y
    # los archivos (CSV, JSON, PNG) nunca se reescriben para un id dado. Sin esta
    # cabecera, Starlette reenvía el archivo completo en cada request (no responde
    # 304), así que cada cambio de grafo re-descargaba 1–2 MB. Con immutable el
    # navegador lo cachea y no vuelve a pedirlo → cambiar/volver de grafo es instantáneo.
    return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/datasets/{dataset_id}/products/search", response_model=QueryResponse)
def search_products(dataset_id: str, q: str, limit: int = 10) -> QueryResponse:
    return _query(lambda: queries.search_products(dataset_id, q, limit))



@router.get("/datasets/{dataset_id}/paths/client-to-supplier", response_model=QueryResponse)
def client_to_supplier(dataset_id: str, client: str, supplier: str) -> QueryResponse:
    return _query(lambda: queries.client_supplier_path(dataset_id, client, supplier))


@router.get("/datasets/{dataset_id}/paths/weighted", response_model=QueryResponse)
def weighted_path(dataset_id: str, source: str, target: str, graph_type: str = "business") -> QueryResponse:
    return _query(lambda: queries.weighted_connection(dataset_id, source, target, graph_type))


@router.get("/datasets/{dataset_id}/paths/logistics-a-star", response_model=QueryResponse)
def logistics_a_star(dataset_id: str, client: str, supplier: str) -> QueryResponse:
    return _query(lambda: queries.logistics_a_star(dataset_id, client, supplier))


@router.get("/datasets/{dataset_id}/network/critical-nodes", response_model=QueryResponse)
def critical_nodes(dataset_id: str, graph_type: str = "business", limit: int = 20) -> QueryResponse:
    return _query(lambda: queries.critical_nodes(dataset_id, graph_type, limit))


@router.post("/datasets/{dataset_id}/supply/min-cost-flow", response_model=QueryResponse)
def min_cost_flow(dataset_id: str, body: MinCostFlowRequest) -> QueryResponse:
    items = [item.model_dump() for item in body.items]
    return _query(lambda: queries.min_cost_supply(dataset_id, items))


@router.get("/datasets/{dataset_id}/mst", response_model=QueryResponse)
def mst(dataset_id: str, graph_type: str = "business", limit: int = 60) -> QueryResponse:
    return _query(lambda: queries.mst_kruskal(dataset_id, graph_type, limit))


@router.get("/datasets/{dataset_id}/components", response_model=QueryResponse)
def components(dataset_id: str, graph_type: str = "business", limit: int = 20) -> QueryResponse:
    return _query(lambda: queries.graph_components(dataset_id, graph_type, limit))


@router.get("/datasets/{dataset_id}/suppliers/{supplier_id}/substitutes", response_model=QueryResponse)
def supplier_substitutes(dataset_id: str, supplier_id: str) -> QueryResponse:
    return _query(lambda: queries.supplier_substitutes(dataset_id, supplier_id))


@router.post("/datasets/{dataset_id}/budget/optimize", response_model=QueryResponse)
def budget_optimize(dataset_id: str, body: BudgetRequest) -> QueryResponse:
    items = [item.model_dump() for item in body.items]
    return _query(lambda: queries.optimize_budget(dataset_id, body.budget, items))


@router.get("/datasets/{dataset_id}/offers/best-savings", response_model=QueryResponse)
def offers(dataset_id: str, limit: int = 20) -> QueryResponse:
    return _query(lambda: queries.best_savings(dataset_id, limit))


@router.get("/datasets/{dataset_id}/products/{product_id}/cross-sell", response_model=QueryResponse)
def cross_sell(dataset_id: str, product_id: str, limit: int = 10) -> QueryResponse:
    return _query(lambda: queries.cross_sell(dataset_id, product_id, limit))


@router.get("/datasets/{dataset_id}/products/{product_id}/co-occurrence", response_model=QueryResponse)
def product_co_occurrence(dataset_id: str, product_id: str, graph_type: str = "sales", limit: int = 15) -> QueryResponse:
    """
    Market Basket Analysis: qué productos aparecen en el MISMO DOCUMENTO que uno dado.
    Diferencia con cross-sell: cross-sell es histórico (cliente compró A y B en cualquier momento).
    Co-occurrence es operativo (A y B estaban en la misma factura/comprobante).
    """
    return _query(lambda: queries.product_co_occurrence(dataset_id, product_id, graph_type, limit))


@router.get("/datasets/{dataset_id}/graph/summary")
def graph_summary(dataset_id: str) -> dict:
    try:
        return queries.graph_summary(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _query(call):
    try:
        return call()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error ejecutando consulta: {exc}") from exc
