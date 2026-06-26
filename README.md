# API Comercial de Envases

Backend FastAPI para cargar archivos de productos, ventas y compras de una empresa
de envases de vidrio/plastico, generar artefactos reproducibles y consultar
algoritmos defendibles para el curso.

## Ejecutar

```bash
cd Final/envases_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Tambien se puede arrancar con:

```bash
python run.py
```

Abrir:

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

## Docker

Construir y correr con Docker:

```bash
docker build -t envases-api .
docker run --rm -p 8000:8000 -v "$(pwd)/storage/datasets:/app/storage/datasets" envases-api
```

O con Docker Compose:

```bash
docker compose up --build
```

Si un frontend web vive en otro dominio, configura CORS con `ALLOWED_ORIGINS`:

```bash
docker run --rm -p 8000:8000 \
  -e ALLOWED_ORIGINS="https://tu-frontend.com,http://localhost:3000" \
  -v "$(pwd)/storage/datasets:/app/storage/datasets" \
  envases-api
```

Desde un frontend, la URL base seria:

```text
http://localhost:8000
```

En produccion deberia ser una URL HTTPS publica, por ejemplo:

```text
https://api.tu-dominio.com
```

## Demo Principal

En Swagger ejecutar `POST /datasets` con:

- `productos`: `data/base/productos.csv`
- `ventas`: `data/base/ventas.csv`
- `compras`: `data/base/items_compras.csv`

La respuesta devuelve un `dataset_id`. Usarlo en los endpoints:

- `GET /datasets/{dataset_id}`
- `GET /datasets/{dataset_id}/products/search?q=frasco gotero ambar`
- `GET /datasets/{dataset_id}/clients/{client_id}/profile`
- `GET /datasets/{dataset_id}/paths/client-to-supplier?client=ODONTOLOGIA&supplier=QUESITO`
- `GET /datasets/{dataset_id}/products/5004/substitutes`
- `POST /datasets/{dataset_id}/budget/optimize`
- `GET /datasets/{dataset_id}/offers/best-savings`
- `POST /datasets/{dataset_id}/purchase/optimize`

## Algoritmos

Nucleo de sustentacion:

- BFS y BFS bidireccional para conexiones en `G_business`.
- Programacion dinamica tipo knapsack para presupuesto.
- Bellman-Ford para mejores ahorros historicos.
- UFDS/componentes conexos para familias y sustitutos.

Extra practico:

- Min-cost flow para compra multi-producto/proveedor.

No se usa Streamlit ni Gemini en esta version.

## Artefactos

Cada carga crea una carpeta:

```text
storage/datasets/{dataset_id}/
```

Alli se guardan CSV/JSON limpios, grafos, metricas, familias, opciones de compra
y resultados de Bellman-Ford. Si un artefacto no se puede generar, queda listado
en `dataset_summary.json` con la razon.

## Fixtures

`data/fixtures/` contiene:

- `completo`: muestra valida.
- `columnas_renombradas`: prueba deteccion de esquema por aliases.
- `incompleto`: prueba degradacion/rechazo por insuficiencia.
