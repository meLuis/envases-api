from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArtifactStatus(BaseModel):
    name: str
    kind: str
    generated: bool = True
    reason: str | None = None


class DatasetSummary(BaseModel):
    dataset_id: str
    status: str
    row_counts: dict[str, int] = Field(default_factory=dict)
    generated: list[ArtifactStatus] = Field(default_factory=list)
    omitted: list[ArtifactStatus] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    ok: bool = True
    answer: str = ""
    algorithm: str = ""
    table: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class BudgetItem(BaseModel):
    product_id: str
    quantity: float = 1.0
    value: float | None = None


class BudgetRequest(BaseModel):
    budget: float
    items: list[BudgetItem]


class PurchaseItem(BaseModel):
    product_id: str
    quantity: float


class PurchaseOptimizeRequest(BaseModel):
    items: list[PurchaseItem]

