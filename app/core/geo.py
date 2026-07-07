from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any

import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en línea recta (gran círculo) entre dos coordenadas, en km.

    Se usa para ponderar distancias geograficas aproximadas entre entidades y
    puntos de ruta dentro de la capa logistica.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = map(radians, (lat1, lon1, lat2, lon2))
    d_lat = lat2_r - lat1_r
    d_lon = lon2_r - lon1_r
    a = sin(d_lat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def _to_coord(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    return num


def parse_lat_lon(lat: Any, lon: Any) -> tuple[float, float] | None:
    """Valida un par (lat, lon). Devuelve None si falta o está fuera de rango."""
    plat = _to_coord(lat)
    plon = _to_coord(lon)
    if plat is None or plon is None:
        return None
    if not (-90.0 <= plat <= 90.0) or not (-180.0 <= plon <= 180.0):
        return None
    if plat == 0.0 and plon == 0.0:
        return None
    return (plat, plon)
