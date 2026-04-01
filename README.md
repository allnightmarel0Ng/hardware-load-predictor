# hardware-load-predictor

Predicts CPU, RAM, and network load on compute infrastructure from business
metrics (e.g. orders/min, active users) using machine learning.

## Architecture

Seven internal modules map directly to the diploma project design:

| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Configuration Management | `app/modules/config_manager.py` | ✅ Real |
| 2 | Historical Data Collector | `app/modules/data_collector.py` | 🟡 Prometheus stub |
| 3 | Correlation Analyzer | `app/modules/correlation_analyzer.py` | 🟡 Mocked values |
| 4 | Model Trainer | `app/modules/model_trainer.py` | 🟡 Mock linear model |
| 5 | Forecasting Engine | `app/modules/forecasting_engine.py` | ✅ Real (uses model) |
| 6 | Accuracy Monitor | `app/modules/accuracy_monitor.py` | 🟡 Mock accuracy check |
| 7 | Request Handler (API) | `app/modules/request_handler.py` | ✅ Real REST API |

**Mocked** = correct interface, realistic fake data, easy to swap for real implementation.

## Quickstart

```bash
# 1. Copy env config
cp .env.example .env

# 2. Start Postgres + app
docker compose up --build

# 3. Open interactive docs
open http://localhost:8000/docs
```

## API Endpoints

### Configuration
```
POST   /configs/              — Register a new config (host + business metric formula)
GET    /configs/              — List all configs
GET    /configs/{id}          — Get a single config
PATCH  /configs/{id}          — Update a config
DELETE /configs/{id}          — Delete a config
```

### Training
```
POST   /configs/{id}/train/         — Collect data, analyze, train model
GET    /configs/{id}/train/models   — List trained models for a config
```

### Forecasting
```
POST   /configs/{id}/forecast/  — Predict CPU/RAM/Network for a business metric value
```

## Example Workflow

```bash
# 1. Create a config
curl -X POST http://localhost:8000/configs/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "orders-to-infra",
    "host": "prometheus.internal",
    "port": 9090,
    "business_metric_name": "orders_per_minute",
    "business_metric_formula": "sum(rate(orders_total[1m]))"
  }'
# → {"id": 1, "name": "orders-to-infra", ...}

# 2. Train a model
curl -X POST http://localhost:8000/configs/1/train/ \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 30}'
# → {"message": "Model trained successfully.", "model_id": 1, "status": "ready"}

# 3. Get a forecast
curl -X POST http://localhost:8000/configs/1/forecast/ \
  -H "Content-Type: application/json" \
  -d '{"business_metric_value": 1500.0}'
# → {"predicted_cpu_percent": 45.2, "predicted_ram_gb": 12.1, "predicted_network_mbps": 87.3, ...}
```

## Running Tests

```bash
pip install -r requirements.txt
pytest -v
```

Test coverage:
- `tests/test_config_manager.py`      — Module 1 unit tests
- `tests/test_data_collector.py`      — Module 2 unit tests
- `tests/test_correlation_and_trainer.py` — Modules 3 & 4 unit tests
- `tests/test_forecasting_engine.py`  — Module 5 unit tests
- `tests/test_api.py`                 — Full integration tests (all endpoints)

## Replacing Mocks with Real ML

### Module 2 — Connect to real Prometheus
In `data_collector.py`, uncomment the real implementation in `_query_prometheus()`:
```python
url = f"http://{host}:{port}/api/v1/query_range"
resp = httpx.get(url, params={"query": formula, "start": ..., "end": ..., "step": step_seconds})
```

### Module 3 — Real cross-correlation
In `correlation_analyzer.py`, replace the mock block in `analyze()` with:
```python
import numpy as np
def _cross_correlate(x, y, max_lag=60):
    best_lag, best_r = 0, 0.0
    for lag in range(0, max_lag + 1):
        r = float(np.corrcoef(x[lag:], y[:len(x)-lag])[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_lag, best_r
```

### Module 4 — Real sklearn model
In `model_trainer.py`, replace `_fit_mock_model()` with:
```python
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100))
X = [[p["value"]] for p in bundle.business]
y = [[c["value"], r["value"], n["value"]]
     for c, r, n in zip(bundle.cpu, bundle.ram, bundle.network)]
model.fit(X, y)
joblib.dump(model, artifact_path)
```

## Project Structure

```
hardware-load-predictor/
├── app/
│   ├── core/
│   │   ├── config.py          # Settings (pydantic-settings)
│   │   └── database.py        # SQLAlchemy engine + session
│   ├── models/
│   │   └── db_models.py       # ORM: ForecastingConfig, TrainedModel, ForecastResult
│   ├── schemas/
│   │   └── schemas.py         # Pydantic request/response models
│   ├── modules/
│   │   ├── config_manager.py      # Module 1
│   │   ├── data_collector.py      # Module 2
│   │   ├── correlation_analyzer.py # Module 3
│   │   ├── model_trainer.py       # Module 4
│   │   ├── forecasting_engine.py  # Module 5
│   │   ├── accuracy_monitor.py    # Module 6
│   │   └── request_handler.py     # Module 7
│   └── main.py                # FastAPI app entrypoint
├── tests/
│   ├── conftest.py
│   ├── test_config_manager.py
│   ├── test_data_collector.py
│   ├── test_correlation_and_trainer.py
│   ├── test_forecasting_engine.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```
