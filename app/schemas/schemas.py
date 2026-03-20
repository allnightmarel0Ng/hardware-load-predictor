from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ── Forecasting Config ────────────────────────────────────────────────────────

class ForecastingConfigCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, examples=["orders-to-cpu"])
    host: str = Field(..., examples=["prometheus.internal"])
    port: int = Field(default=9090, ge=1, le=65535)
    business_metric_name: str = Field(..., examples=["orders_per_minute"])
    business_metric_formula: str = Field(
        ..., examples=["sum(rate(orders_total[1m]))"]
    )


class ForecastingConfigUpdate(BaseModel):
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    business_metric_name: str | None = None
    business_metric_formula: str | None = None


class ForecastingConfigRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
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


# ── Forecast ──────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    business_metric_value: float = Field(
        ..., gt=0, examples=[1500.0], description="Expected value of the business metric"
    )


class ForecastResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    config_id: int
    model_id: int
    business_metric_value: float
    predicted_cpu_percent: float
    predicted_ram_gb: float
    predicted_network_mbps: float
    created_at: datetime


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
