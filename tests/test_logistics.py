from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services import queries
from app.services.pipeline import build_dataset_from_paths


# Familias de producto (como pide el dataset sintético): frasco, tapa, gotero, pote, atomizador.
PRODUCTS = [
    ("1001", "FRASCO VIDRIO AMBAR 30ML", "FRASCOS", 1.20, 2.50),
    ("1002", "FRASCO VIDRIO AMBAR 50ML", "FRASCOS", 1.50, 3.00),
    ("1003", "FRASCO PLASTICO 100ML", "FRASCOS", 0.80, 1.80),
    ("2001", "TAPA ROSCA NEGRA 28MM", "TAPAS", 0.20, 0.50),
    ("2002", "TAPA DISPENSADORA BLANCA 24MM", "TAPAS", 0.35, 0.80),
    ("3001", "GOTERO VIDRIO 30ML", "GOTEROS", 0.60, 1.40),
    ("3002", "GOTERO PLASTICO 20ML", "GOTEROS", 0.40, 1.00),
    ("4001", "POTE CREMA 50G", "POTES", 0.90, 2.00),
    ("4002", "POTE CREMA 100G", "POTES", 1.10, 2.40),
    ("5001", "ATOMIZADOR SPRAY 60ML", "ATOMIZADORES", 1.30, 2.80),
    ("5002", "ATOMIZADOR SPRAY 100ML", "ATOMIZADORES", 1.60, 3.20),
    ("5003", "ATOMIZADOR ROLL-ON 10ML", "ATOMIZADORES", 0.70, 1.60),
]

# Clientes sintéticos con coordenadas (distritos de Lima).
CLIENTS = [
    ("20100000001", "COSMETICOS DEL SUR SAC", -12.1200, -77.0200),
    ("20100000002", "LABORATORIO ANDINO EIRL", -12.0500, -77.0500),
    ("20100000003", "BOTICA NATURAL SRL", -12.0000, -76.9800),
    ("20100000004", "DISTRIBUIDORA NORTE SAC", -11.9800, -77.0700),
]

# Proveedores sintéticos con coordenadas y catálogos que se solapan.
SUPPLIERS = [
    ("20200000001", "ENVASES LIMA SAC", -12.0400, -77.0300, ["1001", "1002", "2001", "3001", "5001"]),
    ("20200000002", "PLASTICOS PERU EIRL", -12.0800, -77.0100, ["1002", "1003", "2001", "2002", "4001"]),
    ("20200000003", "VIDRIOS ANDINOS SAC", -12.1500, -76.9900, ["1001", "3001", "3002", "5001", "5002"]),
    ("20200000004", "PACKAGING TOTAL SRL", -11.9500, -77.0900, ["4001", "4002", "5002", "5003", "2002"]),
]


def _write_synthetic(base: Path) -> None:
    prod_rows = [
        {
            "CODIGO INTERNO": pid,
            "DESCRIPCIÓN": name,
            "CÓDIGO UNIDAD DE MEDIDA": "NIU",
            "DESCRIPCIÓN DE CATEGORÍA": cat,
            "PRECIO COMPRA UNITARIO (CON IGV)": buy,
            "VALOR VENTA UNITARIO (SIN IGV)": sell,
            "STOCK ACTUAL DISPONIBLE": 500,
        }
        for pid, name, cat, buy, sell in PRODUCTS
    ]
    pd.DataFrame(prod_rows).to_csv(base / "productos.csv", index=False, encoding="utf-8-sig")

    name_of = {pid: name for pid, name, *_ in PRODUCTS}
    sell_of = {pid: sell for pid, _, _, _, sell in PRODUCTS}

    # Cada cliente compra 3 productos; algunos comparten documento (co-ocurrencia).
    sale_rows = []
    for doc, (ruc, cli, lat, lon) in enumerate(CLIENTS, start=1):
        basket = [PRODUCTS[i][0] for i in range((doc - 1) % 3, ((doc - 1) % 3) + 3)]
        for pid in basket:
            qty = 4
            sale_rows.append(
                {
                    "FECHA DE EMISIÓN": "01/03/2025",
                    "TIPO": "01",
                    "SERIE": "F001",
                    "NÚMERO": doc,
                    "DOC ENTIDAD NÚMERO": ruc,
                    "DENOMINACIÓN ENTIDAD": cli,
                    "CÓDIGO": pid,
                    "DESCRIPCIÓN": name_of[pid],
                    "CANTIDAD": qty,
                    "PRECIO UNITARIO": sell_of[pid],
                    "SUBTOTAL": round(qty * sell_of[pid], 2),
                    "TOTAL": round(qty * sell_of[pid] * 1.18, 2),
                    "ANULADO": "NO",
                    "LATITUD": lat,
                    "LONGITUD": lon,
                }
            )
    pd.DataFrame(sale_rows).to_csv(base / "ventas.csv", index=False, encoding="utf-8-sig")

    buy_of = {pid: buy for pid, _, _, buy, _ in PRODUCTS}
    purchase_rows = []
    for doc, (ruc, sup, lat, lon, catalog) in enumerate(SUPPLIERS, start=1):
        for offset, pid in enumerate(catalog):
            qty = 50 + offset * 10
            # cada proveedor vende con un pequeño diferencial de costo
            unit = round(buy_of[pid] * (0.9 + 0.05 * doc), 4)
            purchase_rows.append(
                {
                    "FECHA DE EMISIÓN": "02/03/2025",
                    "SERIE": "C001",
                    "NÚMERO": 1000 + doc,
                    "DOC ENTIDAD NÚMERO": ruc,
                    "DENOMINACIÓN ENTIDAD": sup,
                    "CÓDIGO": pid,
                    "DESCRIPCIÓN": name_of[pid],
                    "CANTIDAD": qty,
                    "PRECIO UNITARIO": unit,
                    "SUBTOTAL": round(qty * unit, 2),
                    "TOTAL": round(qty * unit * 1.18, 2),
                    "LATITUD": lat,
                    "LONGITUD": lon,
                }
            )
    pd.DataFrame(purchase_rows).to_csv(base / "compras.csv", index=False, encoding="utf-8-sig")


def test_synthetic_pipeline_generates_graphs_and_runs_optimizers(tmp_path: Path) -> None:
    _write_synthetic(tmp_path)
    summary = build_dataset_from_paths(
        tmp_path / "productos.csv",
        tmp_path / "ventas.csv",
        tmp_path / "compras.csv",
    )
    assert summary.status == "ready"
    generated = {item.name for item in summary.generated}
    assert "logistics_edges.csv" in generated
    assert "logistics_nodes.csv" in generated

    dataset_id = summary.dataset_id

    gs = queries.graph_summary(dataset_id)
    assert gs["logistics_available"] is True


    # Min-cost flow: demanda de un producto que varios proveedores cubren.
    mcf = queries.min_cost_supply(dataset_id, [{"product_id": "1002", "quantity": 60}])
    assert mcf.ok
    assert mcf.metrics["served_units"] > 0
    assert mcf.metrics["total_cost"] > 0


def test_legacy_dataset_without_coords_skips_logistics() -> None:
    base = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "completo"
    summary = build_dataset_from_paths(base / "productos.csv", base / "ventas.csv", base / "compras.csv")
    generated = {item.name for item in summary.generated}
    assert "logistics_edges.csv" not in generated
