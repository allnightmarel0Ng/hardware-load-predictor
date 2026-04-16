from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ── Server Groups ─────────────────────────────────────────────────────────────

class ServerGroupCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, examples=["api-cluster-prod"])
    description: str | None = Field(default=None)
    business_metric_name: str = Field(..., examples=["orders_per_minute"])
    business_metric_formula: str = Field(..., examples=["sum(rate(orders_total[1m]))"])
    metrics_host: str = Field(..., examples=["prometheus.internal"])
    metrics_port: int = Field(default=9090, ge=1, le=65535)


class ServerGroupUpdate(BaseModel):
    description: str | None = None
    business_metric_name: str | None = None
    business_metric_formula: str | None = None
    metrics_host: str | None = None
    metrics_port: int | None = Field(default=None, ge=1, le=65535)


class ServerRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    group_id: int
    name: str
    host: str
    port: int
    tags: dict | None
    is_active: bool
    created_at: datetime


class ServerGroupRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str | None
    business_metric_name: str
    business_metric_formula: str
    metrics_host: str
    metrics_port: int
    created_at: datetime
    updated_at: datetime
    servers: list[ServerRead] = []


# ── Servers ───────────────────────────────────────────────────────────────────

class ServerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, examples=["api-node-1"])
    host: str = Field(..., examples=["10.0.1.10"])
    port: int = Field(default=9090, ge=1, le=65535)
    tags: dict | None = Field(default=None)


class ServerUpdate(BaseModel):
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    tags: dict | None = None
    is_active: bool | None = None


# ── Cluster Forecast ──────────────────────────────────────────────────────────

class ServerForecastRead(BaseModel):
    server_id: int
    server_name: str
    host: str
    config_id: int
    predicted_cpu_percent:     float
    predicted_ram_gb:          float
    predicted_ram_percent:     float
    predicted_network_mbps:    float
    predicted_disk_io_percent: float
    forecast_result_id: int


class ClusterForecastRequest(BaseModel):
    business_metric_value: float = Field(..., gt=0, examples=[5000.0])


class ClusterForecastResponse(BaseModel):
    group_id: int
    group_name: str
    business_metric_value: float
    n_servers: int
    servers: list[ServerForecastRead]
    # Cluster-level aggregates
    cluster_cpu_avg_percent:     float
    cluster_ram_total_gb:        float
    cluster_ram_avg_percent:     float
    cluster_network_total_mbps:  float
    cluster_disk_avg_io_percent: float
    skipped_servers: list[str]


# ── Provision ─────────────────────────────────────────────────────────────────

class ProvisionResponse(BaseModel):
    group_id: int
    configs_created: int
    config_ids: list[int]
    message: str


# ── Forecasting Config ────────────────────────────────────────────────────────

class ForecastingConfigCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, examples=["orders-to-cpu"])
    host: str = Field(..., examples=["prometheus.internal"])
    port: int = Field(default=9090, ge=1, le=65535)
    business_metric_name: str = Field(..., examples=["orders_per_minute"])
    business_metric_formula: str = Field(..., examples=["sum(rate(orders_total[1m]))"])
    instance_label: str | None = Field(
        default=None,
        examples=["gateway"],
        description="Prometheus instance label. Replaces INSTANCE_PLACEHOLDER in system metric queries.",
    )
    quality_metric_overrides: dict[str, str] | None = Field(
        default=None,
        examples=[{"cpu": "r2", "ram_gb": "rel_mae", "net": "r2+mape"}],
        description=(
            "Override the primary quality metric per target. "
            "Keys: cpu | ram_gb | ram_pct | net | disk. "
            "Values: r2 | mape | rel_mae | mae | r2+mape | auto. "
            "Default (auto) selects the metric based on the target nature: "
            "r2 for CPU, rel_mae for RAM, r2+mape for Net/Disk."
        ),
    )


class ForecastingConfigUpdate(BaseModel):
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    business_metric_name: str | None = None
    business_metric_formula: str | None = None
    instance_label: str | None = None
    quality_metric_overrides: dict[str, str] | None = None


class ForecastingConfigRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    server_id: int | None
    host: str
    port: int
    business_metric_name: str
    business_metric_formula: str
    instance_label: str | None
    quality_metric_overrides: dict[str, str] | None
    created_at: datetime
    updated_at: datetime


# ── Trained Model ─────────────────────────────────────────────────────────────

class TargetQualityInfo(BaseModel):
    """
    Quality report for one system metric target.

    The primary quality metric is chosen based on the nature of each metric:
      CPU         → R²  (dynamic, directly driven by load)
      RAM GB/PCT  → rel_mae = MAE/mean  (inertial, low variance — R² unreliable)
      Network     → R² + MAPE combined  (moderately dynamic, often noisy)
      Disk        → R² + MAPE combined  (inertial, weak business coupling)

    Grades follow universal thresholds (see API docs):
      CPU:  excellent R²≥0.95 MAPE≤5% | good R²≥0.85 MAPE≤10% | satisfactory R²≥0.70 MAPE≤15%
      RAM:  excellent relMAE≤0.01 | good ≤0.03 | satisfactory ≤0.05
      Net:  excellent R²≥0.90 MAPE≤10% | good R²≥0.75 MAPE≤20% | satisfactory R²≥0.50 MAPE≤30%
      Disk: excellent R²≥0.85 MAPE≤15% | good R²≥0.65 MAPE≤25% | satisfactory R²≥0.40 MAPE≤40%
    """
    # ── Correlation info ──────────────────────────────────────────────────────
    lag_steps:   int   = Field(..., description="Detected lag in steps (from correlation analysis on training split)")
    lag_minutes: int   = Field(..., description="Detected lag in minutes")
    r_star:      float = Field(..., description="Best correlation coefficient r* = max(|Pearson|, |Spearman|) at optimal lag")
    rel_std:     float = Field(..., description="Coefficient of variation (std/mean) — indicates metric variance; <0.05 = nearly constant")

    # ── Adaptive model chosen ─────────────────────────────────────────────────
    model_type: str = Field(
        ...,
        description=(
            "Algorithm selected based on r* and rel_std: "
            "xgboost (r*≥0.7) | gbr (r*≥0.5) | ridge (r*≥0.3) | mean_baseline (rel_std<0.05)"
        )
    )

    # ── Primary quality metric ────────────────────────────────────────────────
    quality_metric: str = Field(
        ...,
        description=(
            "Primary evaluation metric chosen for this target's nature: "
            "r2 (CPU — dynamic) | rel_mae (RAM — inertial, low variance) | "
            "r2+mape (Net, Disk — moderately dynamic)"
        )
    )
    quality_value: float = Field(
        ...,
        description="Value of the primary quality metric on the test split"
    )
    grade: str = Field(
        ...,
        description="Qualitative grade based on universal thresholds: excellent | good | satisfactory | poor"
    )
    quality_reasoning: str = Field(
        ...,
        description="Human-readable explanation of why this metric and grade were chosen"
    )

    # ── All standard metrics (always computed) ────────────────────────────────
    r2:      float | None = Field(None, description="R² on test split (None for mean_baseline)")
    mae:     float | None = Field(None, description="MAE on test split")
    mape:    float | None = Field(None, description="MAPE % on test split")
    rel_mae: float | None = Field(None, description="Relative MAE = MAE/mean (most meaningful for RAM)")


class TrainedModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    config_id: int
    version: int
    algorithm: str
    status: str
    parameters: dict | None
    metrics: dict | None
    lag_minutes: int | None
    trained_at: datetime | None
    created_at: datetime
    # Per-target quality report (populated from parameters when available)
    per_target: dict[str, TargetQualityInfo] | None = Field(
        None,
        description=(
            "Per-target quality report. Keys: cpu, ram_gb, ram_pct, net, disk. "
            "Includes detected lag, correlation strength, chosen algorithm, "
            "primary quality metric and its grade."
        )
    )


# ── Forecast (single-step with prediction intervals) ──────────────────────────

class ForecastRequest(BaseModel):
    business_metric_value: float = Field(
        ..., gt=0, examples=[1500.0],
        description="Expected value of the business metric"
    )


class ForecastResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    config_id: int
    model_id: int
    business_metric_value: float
    # Point predictions (five targets)
    predicted_cpu_percent:     float
    predicted_ram_gb:          float
    predicted_ram_percent:     float
    predicted_network_mbps:    float
    predicted_disk_io_percent: float
    # 80% prediction intervals (None if model trained on < 50 samples)
    lower_cpu_percent:         float | None
    lower_ram_gb:              float | None
    lower_ram_percent:         float | None
    lower_network_mbps:        float | None
    lower_disk_io_percent:     float | None
    upper_cpu_percent:         float | None
    upper_ram_gb:              float | None
    upper_ram_percent:         float | None
    upper_network_mbps:        float | None
    upper_disk_io_percent:     float | None
    created_at: datetime
    created_at: datetime


# ── Forecast horizon (multi-step) ─────────────────────────────────────────────

class HorizonStep(BaseModel):
    business_metric_value: float = Field(..., gt=0, examples=[1500.0])
    minutes_ahead: int = Field(
        ..., ge=0, le=1440,
        description="Minutes into the future this value is expected at (0 = now)"
    )


class HorizonForecastRequest(BaseModel):
    steps: list[HorizonStep] = Field(
        ..., min_length=1, max_length=288,
        description=(
            "Ordered list of expected business metric values at future timestamps. "
            "Max 288 steps (24 h at 5-minute resolution)."
        ),
    )


class HorizonStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    step: int
    minutes_ahead: int
    business_metric_value: float
    predicted_cpu_percent: float
    predicted_ram_gb: float
    predicted_network_mbps: float
    lower_cpu_percent: float | None
    lower_ram_gb: float | None
    lower_network_mbps: float | None
    upper_cpu_percent: float | None
    upper_ram_gb: float | None
    upper_network_mbps: float | None


class HorizonForecastResponse(BaseModel):
    config_id: int
    model_id: int
    n_steps: int
    steps: list[HorizonStepResponse]


# ── Training trigger ──────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    lookback_days: int = Field(
        default=30, ge=1, le=365,
        description="How many days of historical data to use for training"
    )


class TrainResponse(BaseModel):
    message: str
    model_id: int
    status: str


# ── Accuracy Monitor ──────────────────────────────────────────────────────────

class ModelEvaluationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    model_id: int
    config_id: int
    n_samples: int
    mae_cpu:      float | None
    mae_ram_gb:   float | None
    mae_ram_pct:  float | None
    mae_net:      float | None
    mae_disk:     float | None
    rmse_cpu:     float | None
    rmse_ram_gb:  float | None
    rmse_ram_pct: float | None
    rmse_net:     float | None
    rmse_disk:    float | None
    mape_overall: float | None
    r2_cpu:       float | None
    r2_ram_gb:    float | None
    r2_ram_pct:   float | None
    r2_net:       float | None
    r2_disk:      float | None
    psi_value:    float | None
    psi_level:    str | None
    triggered_retrain: bool
    evaluated_at: datetime


class AccuracyStatusResponse(BaseModel):
    model_id: int
    config_id: int
    config_name: str
    n_evaluations: int
    n_samples_total: int
    latest_evaluation: ModelEvaluationRead | None
    is_healthy: bool
    health_reason: str | None


# ── Training Jobs ─────────────────────────────────────────────────────────────

class TrainJobResponse(BaseModel):
    job_id: int
    config_id: int
    status: str
    message: str


class TrainJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    config_id: int
    model_id: int | None
    status: str
    lookback_days: int
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
