# hardware-load-predictor

Predicts CPU%, RAM (GB), and network (Mbps) load on compute infrastructure
from business-level metrics (orders/min, active users, requests/sec) using
machine learning. Supports single-server and multi-server cluster deployments.

## Quickstart

```bash
cp .env.example .env
docker compose up --build
open http://localhost:8000/docs
```
