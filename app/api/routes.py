from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.domain.schemas import BudgetRequest, PurchaseOptimizeRequest, QueryResponse
from app.services import queries
from app.services.pipeline import PipelineError, build_dataset_from_uploads
from app.storage.repository import list_artifact_files, require_dataset_dir, list_all_datasets, delete_dataset


router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    return {
        "ok": True,
        "service": "envases_backend",
        "model": request.app.title,
        "version": request.app.version,
    }


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
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artefacto no encontrado.")
    return FileResponse(path)


@router.get("/datasets/{dataset_id}/products/search", response_model=QueryResponse)
def search_products(dataset_id: str, q: str, limit: int = 10) -> QueryResponse:
    return _query(lambda: queries.search_products(dataset_id, q, limit))



@router.get("/datasets/{dataset_id}/paths/client-to-supplier", response_model=QueryResponse)
def client_to_supplier(dataset_id: str, client: str, supplier: str) -> QueryResponse:
    return _query(lambda: queries.client_supplier_path(dataset_id, client, supplier))


@router.get("/datasets/{dataset_id}/paths/weighted", response_model=QueryResponse)
def weighted_path(dataset_id: str, source: str, target: str) -> QueryResponse:
    return _query(lambda: queries.weighted_connection(dataset_id, source, target))


@router.get("/datasets/{dataset_id}/products/{product_id}/substitutes", response_model=QueryResponse)
def product_substitutes(dataset_id: str, product_id: str) -> QueryResponse:
    return _query(lambda: queries.product_substitutes(dataset_id, product_id))


@router.post("/datasets/{dataset_id}/budget/optimize", response_model=QueryResponse)
def budget_optimize(dataset_id: str, body: BudgetRequest) -> QueryResponse:
    items = [item.model_dump() for item in body.items]
    return _query(lambda: queries.optimize_budget(dataset_id, body.budget, items))


@router.get("/datasets/{dataset_id}/offers/best-savings", response_model=QueryResponse)
def offers(dataset_id: str, limit: int = 20) -> QueryResponse:
    return _query(lambda: queries.best_savings(dataset_id, limit))


@router.post("/datasets/{dataset_id}/purchase/optimize", response_model=QueryResponse)
def purchase_optimize(dataset_id: str, body: PurchaseOptimizeRequest) -> QueryResponse:
    items = [item.model_dump() for item in body.items]
    return _query(lambda: queries.optimize_purchase(dataset_id, items))


@router.get("/datasets/{dataset_id}/supply-chain/risk", response_model=QueryResponse)
def supplier_risk(dataset_id: str) -> QueryResponse:
    return _query(lambda: queries.supplier_risk(dataset_id))


@router.get("/datasets/{dataset_id}/products/{product_id}/cross-sell", response_model=QueryResponse)
def cross_sell(dataset_id: str, product_id: str, limit: int = 10) -> QueryResponse:
    return _query(lambda: queries.cross_sell(dataset_id, product_id, limit))


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
