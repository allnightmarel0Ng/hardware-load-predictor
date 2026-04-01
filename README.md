# hardware-load-predictor

Predicts CPU%, RAM (GB), and network (Mbps) load on compute infrastructure
from business-level metrics (orders/min, active users, requests/sec) using
machine learning. Supports single-server and multi-server cluster deployments.

---

## Module Status

| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Configuration Manager | `app/modules/config_manager.py` | ✅ Real |
| 2 | Historical Data Collector | `app/modules/data_collector.py` | 🟡 Prometheus stub (real HTTP commented in) |
| 3 | Correlation Analyzer | `app/modules/correlation_analyzer.py` | ✅ Real — Pearson + Spearman CCF, lag sweep 0–60 min |
| 4 | Model Trainer | `app/modules/model_trainer.py` | ✅ Real — GradientBoostingRegressor + Ridge fallback |
| 5 | Forecasting Engine | `app/modules/forecasting_engine.py` | ✅ Real — joblib inference, 9-feature vector |
| 6 | Accuracy Monitor | `app/modules/accuracy_monitor.py` | ✅ Real — Prometheus actuals, MAE/RMSE/R², PSI drift |
| 7 | Request Handler (API) | `app/modules/request_handler.py` | ✅ Real REST API |
| + | Job Runner | `app/modules/job_runner.py` | ✅ Real — async ThreadPoolExecutor |
| + | Drift Detector | `app/modules/drift_detector.py` | ✅ Real — PSI (Population Stability Index) |
| + | Server Group Manager | `app/modules/server_group_manager.py` | ✅ Real — multi-server CRUD |
| + | Cluster Forecaster | `app/modules/cluster_forecaster.py` | ✅ Real — aggregate forecast across N servers |

---

## Quickstart

```bash
cp .env.example .env
docker compose up --build
open http://localhost:8000/docs
```

Apply database migrations:
```bash
alembic upgrade head
```

---

## API Reference

### Server Groups (multi-server)
```
POST   /groups/                            — Create a server group (shared business metric)
GET    /groups/                            — List all groups
GET    /groups/{id}                        — Get group with server list
PATCH  /groups/{id}                        — Update group
DELETE /groups/{id}                        — Delete group and all servers

POST   /groups/{id}/servers/               — Add a server to the group
GET    /groups/{id}/servers/               — List servers (?active_only=true)
GET    /groups/{id}/servers/{sid}          — Get a server
PATCH  /groups/{id}/servers/{sid}          — Update server (host, port, tags, is_active)
DELETE /groups/{id}/servers/{sid}          — Remove server

POST   /groups/{id}/provision              — Create ForecastingConfig per active server
POST   /groups/{id}/train/                 — Submit async training jobs for all servers (202)
POST   /groups/{id}/forecast/              — Cluster forecast: one biz value → all servers + aggregates
```

### Configuration (single-server)
```
POST   /configs/                           — Register a config
GET    /configs/                           — List configs
GET    /configs/{id}                       — Get config
PATCH  /configs/{id}                       — Update config
DELETE /configs/{id}                       — Delete config
```

### Training
```
POST   /configs/{id}/train/                — Submit async training job (202 + job_id)
GET    /configs/{id}/train/models          — List trained model versions
GET    /configs/{id}/train/jobs            — List training job history
GET    /jobs/{job_id}                      — Poll training job status
```

### Forecasting
```
POST   /configs/{id}/forecast/             — Predict CPU/RAM/Network for a business metric value
```

### Accuracy & Drift Monitoring
```
GET    /models/{id}/accuracy               — Current accuracy status + health flag
POST   /models/{id}/accuracy/evaluate      — Force immediate evaluation (backfill + PSI + metrics)
GET    /models/{id}/accuracy/history       — Evaluation history (newest first)
```

---

## Typical Workflows

### Single server
```bash
# 1. Register
curl -X POST http://localhost:8000/configs/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "orders-to-infra",
    "host": "prometheus.internal",
    "port": 9090,
    "business_metric_name": "orders_per_minute",
    "business_metric_formula": "sum(rate(orders_total[1m]))"
  }'

# 2. Train (async — returns immediately)
curl -X POST http://localhost:8000/configs/1/train/ \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 30}'
# → {"job_id": 42, "status": "queued", ...}

# 3. Poll job
curl http://localhost:8000/jobs/42
# → {"status": "done", "model_id": 7, "duration_seconds": 14.2, ...}

# 4. Forecast
curl -X POST http://localhost:8000/configs/1/forecast/ \
  -H "Content-Type: application/json" \
  -d '{"business_metric_value": 1500.0}'
# → {"predicted_cpu_percent": 45.2, "predicted_ram_gb": 12.1, "predicted_network_mbps": 87.3, ...}

# 5. Check model health
curl http://localhost:8000/models/7/accuracy
# → {"is_healthy": true, "latest_evaluation": {"r2_cpu": 0.92, "psi_level": "stable", ...}}
```

### Multi-server cluster
```bash
# 1. Create group
curl -X POST http://localhost:8000/groups/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "api-cluster-prod",
    "business_metric_name": "orders_per_minute",
    "business_metric_formula": "sum(rate(orders_total[1m]))",
    "metrics_host": "prometheus.internal",
    "metrics_port": 9090
  }'

# 2. Add servers
curl -X POST http://localhost:8000/groups/1/servers/ \
  -d '{"name": "node-1", "host": "10.0.0.1", "port": 9090}'
curl -X POST http://localhost:8000/groups/1/servers/ \
  -d '{"name": "node-2", "host": "10.0.0.2", "port": 9090}'

# 3. Provision configs + train all
curl -X POST http://localhost:8000/groups/1/provision
curl -X POST http://localhost:8000/groups/1/train/ -d '{"lookback_days": 30}'
# → [{"job_id": 10, ...}, {"job_id": 11, ...}]

# 4. Cluster forecast
curl -X POST http://localhost:8000/groups/1/forecast/ \
  -d '{"business_metric_value": 5000.0}'
# → {
#     "cluster_cpu_avg_percent": 61.4,
#     "cluster_ram_total_gb": 48.2,
#     "cluster_network_total_mbps": 412.0,
#     "servers": [
#       {"server_name": "node-1", "predicted_cpu_percent": 58.1, ...},
#       {"server_name": "node-2", "predicted_cpu_percent": 64.7, ...}
#     ]
#   }
```

---

## ML Pipeline

### Correlation Analysis (Module 3)
- First-differences both series to remove trend
- Sweeps lags 0–60 minutes computing Pearson CCF and Spearman lag-CCF in parallel
- Reports `max(|pearson|, |spearman|)` as combined strength
- Significance threshold: 0.6
- Experimental results (14-day synthetic data): network r=0.785 ✅, CPU r=0.565, RAM r=0.257

### Model Training (Module 4)
- **Features (9):** `biz_lagged`, `roll5_mean/std`, `roll15_mean/std`, `hour_sin/cos`, `dow_sin/cos`
- **Algorithm:** `MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, max_depth=4, lr=0.05, subsample=0.8))`
- **Fallback:** Ridge Regression when < 50 training samples
- **Split:** 80/20 temporal (no shuffling)
- Experimental results: R²\_cpu=0.922, R²\_net=0.986, MAPE=9.36%

### Drift Detection (Module 6 + drift\_detector.py)
Population Stability Index (PSI) monitors whether the business metric distribution has shifted since training:

| PSI | Level | Action |
|-----|-------|--------|
| < 0.10 | Stable | None |
| 0.10 – 0.20 | Moderate | Monitor closely |
| ≥ 0.20 | Significant | Retraining triggered |

At training time the reference distribution is snapshotted into `TrainedModel.parameters["input_distribution"]`. At each evaluation cycle PSI is computed against recent forecasts and persisted in `ModelEvaluation.psi_value`.

### Retraining Triggers
Retraining is triggered (as a new async job) when either:
- Average post-deployment R² < `ACCURACY_THRESHOLD` (default 0.85)
- PSI ≥ 0.20 on the business metric input distribution

---

## Async Training

`POST /configs/{id}/train/` returns HTTP 202 immediately with a `job_id`.
A `ThreadPoolExecutor` (default 4 workers) runs the full pipeline in the background:

```
QUEUED → RUNNING → DONE
                 ↘ FAILED  (traceback stored in error_message, capped at 2000 chars)
```

Poll `GET /jobs/{job_id}` for `status`, `started_at`, `finished_at`, and `duration_seconds`.

---

## Database Schema

Seven tables managed via Alembic migrations:

| Table | Purpose |
|-------|---------|
| `server_groups` | Named cluster with shared business metric formula |
| `servers` | Individual nodes within a group (host, port, tags, is_active) |
| `forecasting_configs` | Per-server ML config; `server_id` FK (nullable for legacy single-server) |
| `trained_models` | Versioned model registry + input\_distribution snapshot for PSI |
| `forecast_results` | Predictions + `actual_*` columns back-filled by accuracy monitor |
| `model_evaluations` | Post-deployment MAE/RMSE/R²/PSI per evaluation run |
| `training_jobs` | Async job lifecycle (QUEUED → RUNNING → DONE/FAILED) |

```bash
alembic upgrade head                          # apply all migrations
alembic revision --autogenerate -m "change"   # generate after model changes
alembic downgrade -1                          # roll back one
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest -v
```

| File | Coverage |
|------|---------|
| `tests/test_config_manager.py` | Module 1 CRUD |
| `tests/test_data_collector.py` | Module 2 stub |
| `tests/test_correlation_and_trainer.py` | Modules 3 + 4 (real math + sklearn) |
| `tests/test_forecasting_engine.py` | Module 5 |
| `tests/test_accuracy_monitor.py` | Module 6 (Prometheus monkeypatched) |
| `tests/test_drift_detector.py` | PSI math |
| `tests/test_job_runner.py` | Async job lifecycle |
| `tests/test_multi_server.py` | ServerGroup/Server CRUD + cluster forecast |
| `tests/test_api.py` | Full integration (all endpoints) |

---

## Connecting Real Prometheus (Module 2)

In `data_collector.py`, the real HTTP call is commented in `_query_prometheus()`.
Uncomment and set in `.env`:

```env
METRICS_SOURCE_URL=http://your-prometheus:9090
```

Override the default PromQL expressions for system metrics:
```env
PROMETHEUS_CPU_QUERY=100 - avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100
PROMETHEUS_RAM_QUERY=(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1073741824
PROMETHEUS_NET_QUERY=sum(irate(node_network_receive_bytes_total[5m])) * 8 / 1048576
```

---

## Evaluation on Real Data

Download pre-processed public cluster traces (CC-BY 4.0, ~800 KB total) and run the evaluation pipeline:

```bash
curl -L "https://zenodo.org/records/14564935/files/machine_usage_days_1_to_8_grouped_300_seconds.csv?download=1" \
     -o scripts/machine_usage_days_1_to_8_grouped_300_seconds.csv

curl -L "https://zenodo.org/records/14564935/files/instance_usage_grouped_300_seconds_month.csv?download=1" \
     -o scripts/instance_usage_grouped_300_seconds_month.csv

python scripts/evaluate_on_real_data.py
```

---

## Project Structure

```
hardware-load-predictor/
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 0001_initial.py              — All 5 original tables
│       └── 0002_multi_server.py         — server_groups, servers, server_id FK
├── app/
│   ├── core/
│   │   ├── config.py                    — Settings (pydantic-settings)
│   │   └── database.py                  — SQLAlchemy engine + session
│   ├── models/
│   │   └── db_models.py                 — ORM: all 7 tables + enums
│   ├── schemas/
│   │   └── schemas.py                   — Pydantic request/response models
│   ├── modules/
│   │   ├── config_manager.py            — Module 1
│   │   ├── data_collector.py            — Module 2
│   │   ├── correlation_analyzer.py      — Module 3
│   │   ├── model_trainer.py             — Module 4
│   │   ├── forecasting_engine.py        — Module 5
│   │   ├── accuracy_monitor.py          — Module 6
│   │   ├── request_handler.py           — Module 7 (all routers)
│   │   ├── job_runner.py                — Async training executor
│   │   ├── drift_detector.py            — PSI drift detection
│   │   ├── server_group_manager.py      — Multi-server CRUD
│   │   └── cluster_forecaster.py        — Cluster forecast aggregation
│   └── main.py                          — FastAPI app + startup/shutdown hooks
├── scripts/
│   └── evaluate_on_real_data.py         — Evaluation on Alibaba/Google traces
├── tests/
│   ├── conftest.py
│   ├── test_config_manager.py
│   ├── test_data_collector.py
│   ├── test_correlation_and_trainer.py
│   ├── test_forecasting_engine.py
│   ├── test_accuracy_monitor.py
│   ├── test_drift_detector.py
│   ├── test_job_runner.py
│   ├── test_multi_server.py
│   └── test_api.py
├── alembic.ini
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── .env.example
```
