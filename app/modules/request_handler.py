"""
Module 7 — Request Handler
FastAPI routers that expose the system's REST API.
"""
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.modules import (
    config_manager,
    data_collector,
    correlation_analyzer,
    model_trainer,
    forecasting_engine,
    accuracy_monitor,
    job_runner,
    server_group_manager,
    cluster_forecaster,
)
from app.schemas.schemas import (
    AccuracyStatusResponse,
    ClusterForecastRequest,
    ClusterForecastResponse,
    ForecastingConfigCreate,
    ForecastingConfigRead,
    ForecastingConfigUpdate,
    ForecastRequest,
    ForecastResponse,
    HorizonForecastRequest,
    HorizonForecastResponse,
    HorizonStepResponse,
    ModelEvaluationRead,
    ProvisionResponse,
    ServerCreate,
    ServerGroupCreate,
    ServerGroupRead,
    ServerGroupUpdate,
    ServerForecastRead,
    ServerRead,
    ServerUpdate,
    TrainedModelRead,
    TrainJobRead,
    TrainJobResponse,
    TrainRequest,
    TrainResponse,
)

# ── Server Group router ───────────────────────────────────────────────────────
groups_router = APIRouter(prefix="/groups", tags=["Server Groups"])


@groups_router.post("/", response_model=ServerGroupRead, status_code=status.HTTP_201_CREATED)
def create_group(data: ServerGroupCreate, db: Session = Depends(get_db)):
    """
    Create a server group — a named cluster sharing one business metric.
    Add servers with POST /groups/{id}/servers/ then train with POST /groups/{id}/train/.
    """
    return server_group_manager.create_group(db, data)


@groups_router.get("/", response_model=list[ServerGroupRead])
def list_groups(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return server_group_manager.list_groups(db, skip=skip, limit=limit)


@groups_router.get("/{group_id}", response_model=ServerGroupRead)
def get_group(group_id: int, db: Session = Depends(get_db)):
    return server_group_manager.get_group(db, group_id)


@groups_router.patch("/{group_id}", response_model=ServerGroupRead)
def update_group(group_id: int, data: ServerGroupUpdate, db: Session = Depends(get_db)):
    return server_group_manager.update_group(db, group_id, data)


@groups_router.delete("/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_group(group_id: int, db: Session = Depends(get_db)):
    server_group_manager.delete_group(db, group_id)


# ── Server sub-resource ───────────────────────────────────────────────────────

@groups_router.post(
    "/{group_id}/servers/",
    response_model=ServerRead,
    status_code=status.HTTP_201_CREATED,
)
def add_server(group_id: int, data: ServerCreate, db: Session = Depends(get_db)):
    """Add an individual server (node) to a group."""
    return server_group_manager.add_server(db, group_id, data)


@groups_router.get("/{group_id}/servers/", response_model=list[ServerRead])
def list_servers(
    group_id: int,
    active_only: bool = False,
    db: Session = Depends(get_db),
):
    return server_group_manager.list_servers(db, group_id, active_only=active_only)


@groups_router.get("/{group_id}/servers/{server_id}", response_model=ServerRead)
def get_server(group_id: int, server_id: int, db: Session = Depends(get_db)):
    return server_group_manager.get_server(db, group_id, server_id)


@groups_router.patch("/{group_id}/servers/{server_id}", response_model=ServerRead)
def update_server(
    group_id: int, server_id: int, data: ServerUpdate, db: Session = Depends(get_db)
):
    return server_group_manager.update_server(db, group_id, server_id, data)


@groups_router.delete(
    "/{group_id}/servers/{server_id}", status_code=status.HTTP_204_NO_CONTENT
)
def remove_server(group_id: int, server_id: int, db: Session = Depends(get_db)):
    server_group_manager.remove_server(db, group_id, server_id)


# ── Provision configs ─────────────────────────────────────────────────────────

@groups_router.post("/{group_id}/provision", response_model=ProvisionResponse)
def provision_group(group_id: int, db: Session = Depends(get_db)):
    """
    Create a ForecastingConfig for every active server in the group
    that doesn't already have one.  Must be called before training.
    """
    configs = server_group_manager.provision_group_configs(db, group_id)
    group   = server_group_manager.get_group(db, group_id)
    return ProvisionResponse(
        group_id=group_id,
        configs_created=len(configs),
        config_ids=[c.id for c in configs],
        message=(
            f"Provisioned {len(configs)} config(s) for group '{group.name}'. "
            "Submit training jobs with POST /groups/{id}/train/."
        ),
    )


# ── Group-level training ──────────────────────────────────────────────────────

@groups_router.post(
    "/{group_id}/train/",
    response_model=list[TrainJobResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
def train_group(
    group_id: int, body: TrainRequest, db: Session = Depends(get_db)
):
    """
    Submit an async training job for every active server in the group.
    Returns a list of job responses (one per server config).
    Each job can be polled individually via GET /jobs/{job_id}.
    """
    from app.models.db_models import ForecastingConfig, Server
    servers = server_group_manager.list_servers(db, group_id, active_only=True)
    if not servers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Group {group_id} has no active servers to train.",
        )

    responses: list[TrainJobResponse] = []
    for server in servers:
        config = db.query(ForecastingConfig).filter_by(server_id=server.id).first()
        if config is None:
            continue  # not yet provisioned — skip silently
        job = job_runner.submit_training_job(config.id, body.lookback_days)
        responses.append(TrainJobResponse(
            job_id=job.id,
            config_id=config.id,
            status=job.status.value,
            message=f"Training queued for server '{server.name}'.",
        ))

    if not responses:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No configs found for group {group_id}. "
                "Run POST /groups/{id}/provision first."
            ),
        )
    return responses


# ── Cluster forecast ──────────────────────────────────────────────────────────

@groups_router.post("/{group_id}/forecast/", response_model=ClusterForecastResponse)
def forecast_cluster(
    group_id: int,
    body: ClusterForecastRequest,
    db: Session = Depends(get_db),
):
    """
    Predict CPU, RAM, and network load for every active server in the group
    from a single business metric value.

    Returns per-server predictions plus cluster-level aggregates:
      - cluster_cpu_avg_percent      (mean across servers)
      - cluster_ram_total_gb         (sum across servers)
      - cluster_network_total_mbps   (sum across servers)

    Servers without a trained model are listed in skipped_servers.
    """
    result = cluster_forecaster.forecast_cluster(
        db, group_id, body.business_metric_value
    )

    if not result.servers and result.skipped_servers:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"No server in group {group_id} has a ready model. "
                f"Skipped: {result.skipped_servers}"
            ),
        )

    return ClusterForecastResponse(
        group_id=result.group_id,
        group_name=result.group_name,
        business_metric_value=result.business_metric_value,
        n_servers=result.n_servers,
        servers=[
            ServerForecastRead(
                server_id=s.server_id,
                server_name=s.server_name,
                host=s.host,
                config_id=s.config_id,
                predicted_cpu_percent=s.predicted_cpu_percent,
                predicted_ram_gb=s.predicted_ram_gb,
                predicted_ram_percent=s.predicted_ram_percent,
                predicted_network_mbps=s.predicted_network_mbps,
                predicted_disk_io_percent=s.predicted_disk_io_percent,
                forecast_result_id=s.forecast_result_id,
            )
            for s in result.servers
        ],
        cluster_cpu_avg_percent=result.cluster_cpu_avg_percent,
        cluster_ram_total_gb=result.cluster_ram_total_gb,
        cluster_ram_avg_percent=result.cluster_ram_avg_percent,
        cluster_network_total_mbps=result.cluster_network_total_mbps,
        cluster_disk_avg_io_percent=result.cluster_disk_avg_io_percent,
        skipped_servers=result.skipped_servers,
    )


# ── Config router ─────────────────────────────────────────────────────────────
config_router = APIRouter(prefix="/configs", tags=["Configuration"])


@config_router.post(
    "/", response_model=ForecastingConfigRead, status_code=status.HTTP_201_CREATED
)
def create_config(data: ForecastingConfigCreate, db: Session = Depends(get_db)):
    """Register a new business-metric → server binding."""
    return config_manager.create_config(db, data)


@config_router.get("/", response_model=list[ForecastingConfigRead])
def list_configs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return config_manager.list_configs(db, skip=skip, limit=limit)


@config_router.get("/{config_id}", response_model=ForecastingConfigRead)
def get_config(config_id: int, db: Session = Depends(get_db)):
    return config_manager.get_config(db, config_id)


@config_router.patch("/{config_id}", response_model=ForecastingConfigRead)
def update_config(
    config_id: int, data: ForecastingConfigUpdate, db: Session = Depends(get_db)
):
    return config_manager.update_config(db, config_id, data)


@config_router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_config(config_id: int, db: Session = Depends(get_db)):
    config_manager.delete_config(db, config_id)


# ── Training router ───────────────────────────────────────────────────────────
train_router = APIRouter(prefix="/configs/{config_id}/train", tags=["Training"])


@train_router.post("/", response_model=TrainJobResponse, status_code=status.HTTP_202_ACCEPTED)
def train(config_id: int, body: TrainRequest, db: Session = Depends(get_db)):
    """
    Submit an async training job.  Returns immediately with a job_id.
    Poll GET /jobs/{job_id} to check progress.
    """
    config_manager.get_config(db, config_id)   # 404 guard
    job = job_runner.submit_training_job(config_id, body.lookback_days)
    return TrainJobResponse(
        job_id=job.id,
        config_id=config_id,
        status=job.status.value,
        message="Training job queued. Poll GET /jobs/{job_id} for status.",
    )




# ── Per-target quality report builder ────────────────────────────────────────

def _build_per_target_info(model_record, overrides: dict | None = None) -> dict | None:
    """
    Build per-target quality report from model parameters and metrics.
    Uses universal thresholds from the grading specification.

    overrides: optional dict mapping target key → quality metric name,
               e.g. {"cpu": "mape", "net": "r2"}. When provided, the
               specified metric is used instead of the auto-selected one.
               Allowed values: r2 | mape | rel_mae | mae | r2+mape | auto.

    Returns None if model was trained with old schema (no per_target in params).
    """
    from app.schemas.schemas import TargetQualityInfo

    params    = model_record.parameters or {}
    metrics   = model_record.metrics    or {}
    pt_params = params.get("per_target")
    if not pt_params:
        return None

    overrides  = overrides or {}
    STEP_SECONDS = 300

    # ── Grade functions with exact thresholds from specification ─────────────

    def _grade_cpu(r2, mape):
        m = mape or 999
        if r2 is None: return "poor"
        if r2 >= 0.95 and m <= 5:   return "excellent"
        if r2 >= 0.85 and m <= 10:  return "good"
        if r2 >= 0.70 and m <= 15:  return "satisfactory"
        return "poor"

    def _grade_ram(rel_mae):
        if rel_mae <= 0.01: return "excellent"
        if rel_mae <= 0.03: return "good"
        if rel_mae <= 0.05: return "satisfactory"
        return "poor"

    def _grade_net(r2, mape):
        m = mape or 999
        if r2 is None: return "poor"
        if r2 >= 0.90 and m <= 10:  return "excellent"
        if r2 >= 0.75 and m <= 20:  return "good"
        if r2 >= 0.50 and m <= 30:  return "satisfactory"
        return "poor"

    def _grade_disk(r2, mape, rel_std):
        m = mape or 999
        if r2 is None:
            # Low variance — use rel_std as proxy for rel_mae
            if rel_std <= 0.05:  return "excellent"
            if rel_std <= 0.10:  return "good"
            if rel_std <= 0.15:  return "satisfactory"
            return "poor"
        if r2 >= 0.85 and m <= 15:  return "excellent"
        if r2 >= 0.65 and m <= 25:  return "good"
        if r2 >= 0.40 and m <= 40:  return "satisfactory"
        return "poor"

    # ── Reasoning templates ───────────────────────────────────────────────────

    def _reasoning_cpu(r2, mape, r_star, lag_min, grade):
        sig = "significant" if r_star >= 0.6 else "weak"
        lag_str = f"{lag_min} min lag" if lag_min > 0 else "no lag"
        return (
            f"CPU is dynamic and directly driven by load. "
            f"Primary metric: R² (target ≥0.85). "
            f"Correlation r*={r_star:.3f} ({sig}), {lag_str}. "
            f"R²={r2:.3f}, MAPE={mape:.1f}% → {grade}."
        )

    def _reasoning_ram(rel_mae, rel_std, r_star, model_type, grade):
        reason = (
            "RAM is inertial and low-variance (rel_std={:.4f} < 0.05). "
            if rel_std < 0.05 else
            "RAM is inertial; R² is unreliable for low-variance series. "
        ).format(rel_std)
        return (
            reason +
            f"Primary metric: rel_mae = MAE/mean (target ≤0.05). "
            f"Correlation r*={r_star:.3f}. "
            f"Model: {model_type}. rel_mae≈{rel_mae:.4f} → {grade}."
        )

    def _reasoning_net(r2, mape, r_star, lag_min, grade):
        sig = "significant" if r_star >= 0.6 else "weak"
        lag_str = f"{lag_min} min lag" if lag_min > 0 else "no lag"
        return (
            f"Network is moderately dynamic with potential spikes. "
            f"Primary metrics: R² (target ≥0.75) + MAPE (target ≤20%). "
            f"Correlation r*={r_star:.3f} ({sig}), {lag_str}. "
            f"R²={r2:.3f}, MAPE={mape:.1f}% → {grade}."
        )

    def _reasoning_disk(r2, mape, r_star, rel_std, grade):
        sig = "significant" if r_star >= 0.6 else "weak (disk weakly coupled to business)"
        if r2 is None:
            return (
                f"Disk is inertial (rel_std={rel_std:.4f}); mean_baseline used. "
                f"Correlation r*={r_star:.3f} ({sig}). "
                f"rel_std used as quality proxy → {grade}."
            )
        return (
            f"Disk is inertial with slow changes. "
            f"Primary metrics: R² (target ≥0.65) + MAPE (target ≤25%). "
            f"Correlation r*={r_star:.3f} ({sig}). "
            f"R²={r2:.3f}, MAPE={mape:.1f}% → {grade}."
        )

    # ── Build per-target report ───────────────────────────────────────────────

    result = {}
    for key in ("cpu", "ram_gb", "ram_pct", "net", "disk"):
        info    = pt_params.get(key, {})
        lag     = info.get("lag", 0)
        r_star  = info.get("r_star", 0.0)
        rel_std = info.get("rel_std", 0.0)
        mtype   = info.get("model_type", "unknown")

        r2   = metrics.get(f"r2_{key}")
        mae  = metrics.get(f"mae_{key}")
        mape = metrics.get(f"mape_{key}")
        lag_min = lag * STEP_SECONDS // 60

        # Compute rel_mae from stored mae and mean_val if available
        mean_val = info.get("mean_val")
        if mae is not None and mean_val and mean_val > 1e-9:
            rel_mae_val = round(mae / mean_val, 4)
        else:
            rel_mae_val = round(rel_std, 4)   # fallback approximation

        # Select primary metric — respect explicit override if provided
        is_low_variance = rel_std < 0.05
        override_metric = overrides.get(key, "auto")
        # Resolve "auto" to the nature-based default
        if override_metric == "auto" or not override_metric:
            if key == "cpu":
                override_metric = "r2"
            elif key in ("ram_gb", "ram_pct") or is_low_variance:
                override_metric = "rel_mae"
            elif key == "net":
                override_metric = "r2+mape"
            else:
                override_metric = "r2+mape"

        # Apply the chosen (or overridden) metric
        if override_metric == "r2":
            quality_metric = "r2"
            quality_value  = round(r2, 4) if r2 is not None else 0.0
            grade          = _grade_cpu(r2, mape)
            reasoning      = _reasoning_cpu(r2 or 0, mape or 0, r_star, lag_min, grade)
            if override_metric != ("r2" if key == "cpu" else "auto"):
                reasoning = f"[Override: r2 requested] " + reasoning

        elif override_metric == "rel_mae":
            quality_metric = "rel_mae"
            quality_value  = rel_mae_val
            grade          = _grade_ram(rel_mae_val)
            reasoning      = _reasoning_ram(rel_mae_val, rel_std, r_star, mtype, grade)
            if key not in ("ram_gb", "ram_pct") and not is_low_variance:
                reasoning = f"[Override: rel_mae requested] " + reasoning

        elif override_metric == "mape":
            quality_metric = "mape"
            quality_value  = round(mape, 4) if mape is not None else 0.0
            # Use CPU thresholds for mape standalone grading
            if mape is None:     grade = "poor"
            elif mape <= 5:      grade = "excellent"
            elif mape <= 10:     grade = "good"
            elif mape <= 15:     grade = "satisfactory"
            else:                grade = "poor"
            reasoning = (
                f"[Override: mape requested] r*={r_star:.3f}, "
                f"MAPE={mape:.2f}% → {grade}."
            )

        elif override_metric == "mae":
            quality_metric = "mae"
            quality_value  = round(mae, 4) if mae is not None else 0.0
            grade = "good"   # MAE has no universal threshold — always informational
            reasoning = (
                f"[Override: mae requested — no universal threshold] "
                f"r*={r_star:.3f}, MAE={mae:.4f}."
            )

        else:   # r2+mape (default for net/disk, or explicit override)
            quality_metric = "r2+mape"
            quality_value  = round(r2, 4) if r2 is not None else rel_std
            if key == "net":
                grade     = _grade_net(r2, mape)
                reasoning = _reasoning_net(r2 or 0, mape or 0, r_star, lag_min, grade)
            else:
                grade     = _grade_disk(r2, mape, rel_std)
                reasoning = _reasoning_disk(r2, mape or 0, r_star, rel_std, grade)
            if override_metric == "r2+mape" and key not in ("net", "disk"):
                reasoning = f"[Override: r2+mape requested] " + reasoning

        result[key] = TargetQualityInfo(
            lag_steps      = lag,
            lag_minutes    = lag_min,
            r_star         = round(r_star, 3),
            rel_std        = round(rel_std, 4),
            model_type     = mtype,
            quality_metric = quality_metric,
            quality_value  = quality_value,
            grade          = grade,
            quality_reasoning = reasoning,
            r2      = round(r2,          4) if r2      is not None else None,
            mae     = round(mae,         4) if mae     is not None else None,
            mape    = round(mape,        4) if mape    is not None else None,
            rel_mae = rel_mae_val,
        )
    return result


def _enrich_model(model_record, config=None):
    """Attach per_target quality info to a TrainedModel ORM object for serialisation."""
    overrides = getattr(config, "quality_metric_overrides", None) if config else None
    model_record.per_target = _build_per_target_info(model_record, overrides=overrides)
    return model_record

@train_router.get("/models", response_model=list[TrainedModelRead])
def list_models(config_id: int, db: Session = Depends(get_db)):
    """List all trained models for a config, newest first.

    Quality metrics are graded using the thresholds from the universal
    specification. Override the primary metric per target by setting
    quality_metric_overrides on the config (PATCH /configs/{id}).
    """
    config = config_manager.get_config(db, config_id)
    models = (
        db.query(model_trainer.TrainedModel)
        .filter_by(config_id=config_id)
        .order_by(model_trainer.TrainedModel.version.desc())
        .all()
    )
    return [_enrich_model(m, config=config) for m in models]


@train_router.get("/jobs", response_model=list[TrainJobRead])
def list_jobs(config_id: int, limit: int = 20, db: Session = Depends(get_db)):
    """List recent training jobs for a config, newest first."""
    config_manager.get_config(db, config_id)  # 404 guard
    return job_runner.list_jobs(config_id, limit=limit)


# ── Job status router ─────────────────────────────────────────────────────────
jobs_router = APIRouter(prefix="/jobs", tags=["Jobs"])


@jobs_router.get("/{job_id}", response_model=TrainJobRead)
def get_job(job_id: int):
    """
    Poll the status of a training job.
    status values: queued → running → done | failed
    """
    job = job_runner.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    return job


# ── Forecast router ───────────────────────────────────────────────────────────
forecast_router = APIRouter(prefix="/configs/{config_id}/forecast", tags=["Forecasting"])


@forecast_router.post("/", response_model=ForecastResponse)
def predict(config_id: int, body: ForecastRequest, db: Session = Depends(get_db)):
    """Predict CPU / RAM / Network load for a given business metric value."""
    config = config_manager.get_config(db, config_id)
    try:
        result = forecasting_engine.forecast(db, config, body.business_metric_value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return result


# ── Accuracy monitor router ───────────────────────────────────────────────────
accuracy_router = APIRouter(prefix="/models", tags=["Accuracy"])


@accuracy_router.get("/{model_id}/accuracy", response_model=AccuracyStatusResponse)
def get_accuracy_status(model_id: int, db: Session = Depends(get_db)):
    """
    Return the current post-deployment accuracy status for a model,
    including R² metrics, PSI drift level, and health flag.
    """
    result = accuracy_monitor.get_accuracy_status(db, model_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found.",
        )
    return result


@accuracy_router.post(
    "/{model_id}/accuracy/evaluate",
    response_model=ModelEvaluationRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def force_evaluate(model_id: int, db: Session = Depends(get_db)):
    """Trigger an immediate accuracy + drift evaluation for a specific model."""
    evaluation = accuracy_monitor.force_evaluate(db, model_id)
    if evaluation is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Model {model_id} is not in READY state or has fewer than "
                f"{accuracy_monitor.MIN_EVAL_SAMPLES} evaluated forecasts."
            ),
        )
    return evaluation


@accuracy_router.get(
    "/{model_id}/accuracy/history",
    response_model=list[ModelEvaluationRead],
)
def get_accuracy_history(model_id: int, limit: int = 20, db: Session = Depends(get_db)):
    """Return evaluation history (newest first) for plotting accuracy drift."""
    from app.models.db_models import ModelEvaluation
    return (
        db.query(ModelEvaluation)
        .filter_by(model_id=model_id)
        .order_by(ModelEvaluation.evaluated_at.desc())
        .limit(limit)
        .all()
    )

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules import (
    config_manager,
    data_collector,
    correlation_analyzer,
    model_trainer,
    forecasting_engine,
    accuracy_monitor,
)
from app.schemas.schemas import (
    AccuracyStatusResponse,
    ForecastingConfigCreate,
    ForecastingConfigRead,
    ForecastingConfigUpdate,
    ForecastRequest,
    ForecastResponse,
    ModelEvaluationRead,
    TrainedModelRead,
    TrainRequest,
    TrainResponse,
)

# ── Config router ─────────────────────────────────────────────────────────────
config_router = APIRouter(prefix="/configs", tags=["Configuration"])


@config_router.post(
    "/", response_model=ForecastingConfigRead, status_code=status.HTTP_201_CREATED
)
def create_config(data: ForecastingConfigCreate, db: Session = Depends(get_db)):
    """Register a new business-metric → server binding."""
    return config_manager.create_config(db, data)


@config_router.get("/", response_model=list[ForecastingConfigRead])
def list_configs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return config_manager.list_configs(db, skip=skip, limit=limit)


@config_router.get("/{config_id}", response_model=ForecastingConfigRead)
def get_config(config_id: int, db: Session = Depends(get_db)):
    return config_manager.get_config(db, config_id)


@config_router.patch("/{config_id}", response_model=ForecastingConfigRead)
def update_config(
    config_id: int, data: ForecastingConfigUpdate, db: Session = Depends(get_db)
):
    return config_manager.update_config(db, config_id, data)


@config_router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_config(config_id: int, db: Session = Depends(get_db)):
    config_manager.delete_config(db, config_id)


# ── Training router ───────────────────────────────────────────────────────────
train_router = APIRouter(prefix="/configs/{config_id}/train", tags=["Training"])


@train_router.post("/", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
def train(config_id: int, body: TrainRequest, db: Session = Depends(get_db)):
    """
    Collect historical data, run correlation analysis, and train a model
    for the given config.
    """
    config = config_manager.get_config(db, config_id)

    # Verify Prometheus is reachable before starting the job
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://{config.host}:{config.port}/-/healthy")
            resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Prometheus at {config.host}:{config.port} is not reachable: {exc}. "
                "Check the host and port in your config."
            ),
        )

    # Use the explicit instance_label from the config for per-server PromQL
    # filtering. Falls back to server.name for cluster-provisioned configs.
    instance_label: str | None = (
        config.instance_label
        or (config.server.name if config.server else None)
    )

    bundle = data_collector.fetch_historical_data(
        host=config.host,
        port=config.port,
        business_formula=config.business_metric_formula,
        lookback_days=body.lookback_days,
        instance_label=instance_label,
        step_seconds=settings.step_seconds,
    )

    report = correlation_analyzer.analyze(bundle, base_step_seconds=settings.step_seconds)
    if report.is_business_constant:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "The business metric series has near-zero variance — it appears "
                "constant. Check your PromQL formula or increase the lookback window."
            ),
        )

    model = model_trainer.train_model(db, config, bundle, report)
    return TrainResponse(
        message="Model trained successfully.",
        model_id=model.id,
        status=model.status.value,
    )


@train_router.get("/models", response_model=list[TrainedModelRead])
def list_models(config_id: int, db: Session = Depends(get_db)):
    """List all trained models for a config."""
    config_manager.get_config(db, config_id)  # 404 guard
    models = (
        db.query(model_trainer.TrainedModel)
        .filter_by(config_id=config_id)
        .order_by(model_trainer.TrainedModel.version.desc())
        .all()
    )
    return models


# ── Forecast router ───────────────────────────────────────────────────────────
forecast_router = APIRouter(prefix="/configs/{config_id}/forecast", tags=["Forecasting"])


@forecast_router.post("/", response_model=ForecastResponse)
def predict(
    config_id: int, body: ForecastRequest, db: Session = Depends(get_db)
):
    """
    Predict CPU / RAM / Network load for a given business metric value.
    """
    config = config_manager.get_config(db, config_id)
    try:
        result = forecasting_engine.forecast(db, config, body.business_metric_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        )
    return result


# ── Accuracy monitor router ───────────────────────────────────────────────────
accuracy_router = APIRouter(prefix="/models", tags=["Accuracy"])


@accuracy_router.get("/{model_id}/accuracy", response_model=AccuracyStatusResponse)
def get_accuracy_status(model_id: int, db: Session = Depends(get_db)):
    """
    Return the current post-deployment accuracy status for a model,
    including the latest evaluation metrics and health flag.
    """
    result = accuracy_monitor.get_accuracy_status(db, model_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found.",
        )
    return result


@accuracy_router.post(
    "/{model_id}/accuracy/evaluate",
    response_model=ModelEvaluationRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def force_evaluate(model_id: int, db: Session = Depends(get_db)):
    """
    Trigger an immediate accuracy evaluation for a specific model,
    rather than waiting for the next scheduled run.
    Fetches pending actuals from Prometheus and computes metrics.
    """
    evaluation = accuracy_monitor.force_evaluate(db, model_id)
    if evaluation is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Model {model_id} is not in READY state or has fewer than "
                f"{accuracy_monitor.MIN_EVAL_SAMPLES} evaluated forecasts."
            ),
        )
    return evaluation


@accuracy_router.get(
    "/{model_id}/accuracy/history",
    response_model=list[ModelEvaluationRead],
)
def get_accuracy_history(
    model_id: int,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """
    Return the full evaluation history for a model (newest first).
    Useful for plotting accuracy drift over time.
    """
    from app.models.db_models import ModelEvaluation
    rows = (
        db.query(ModelEvaluation)
        .filter_by(model_id=model_id)
        .order_by(ModelEvaluation.evaluated_at.desc())
        .limit(limit)
        .all()
    )
    return rows

