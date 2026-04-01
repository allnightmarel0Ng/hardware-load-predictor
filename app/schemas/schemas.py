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
    predicted_cpu_percent: float
    predicted_ram_gb: float
    predicted_network_mbps: float
    forecast_result_id: int


class ClusterForecastRequest(BaseModel):
    business_metric_value: float = Field(..., gt=0, examples=[5000.0])


class ClusterForecastResponse(BaseModel):
    group_id: int
    group_name: str
    business_metric_value: float
    n_servers: int
    servers: list[ServerForecastRead]
    cluster_cpu_avg_percent: float
    cluster_ram_total_gb: float
    cluster_network_total_mbps: float
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


class ForecastingConfigUpdate(BaseModel):
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    business_metric_name: str | None = None
    business_metric_formula: str | None = None


class ForecastingConfigRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    server_id: int | None
    host: str
    port: int
    business_metric_name: str
    business_metric_formula: str
    created_at: datetime
    updated_at: datetime


# ── Trained Model ─────────────────────────────────────────────────────────────

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
    # Point predictions
    predicted_cpu_percent: float
    predicted_ram_gb: float
    predicted_network_mbps: float
    # 80% prediction interval (None if model trained on < 50 samples)
    lower_cpu_percent: float | None
    lower_ram_gb: float | None
    lower_network_mbps: float | None
    upper_cpu_percent: float | None
    upper_ram_gb: float | None
    upper_network_mbps: float | None
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
    mae_cpu: float | None
    mae_ram: float | None
    mae_net: float | None
    rmse_cpu: float | None
    rmse_ram: float | None
    rmse_net: float | None
    mape_overall: float | None
    r2_cpu: float | None
    r2_ram: float | None
    r2_net: float | None
    psi_value: float | None
    psi_level: str | None
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
