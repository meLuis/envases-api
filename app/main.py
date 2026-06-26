from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import ALLOWED_ORIGINS


app = FastAPI(
    title="API Comercial de Envases",
    description="Backend para ingesta, grafos y algoritmos sobre ventas/compras de envases de vidrio y plastico.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
