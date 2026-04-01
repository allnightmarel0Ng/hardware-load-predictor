from datetime import datetime
from sqlalchemy import String, Float, DateTime, JSON, ForeignKey, Enum as SAEnum, Integer, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.database import Base


class ModelStatus(str, enum.Enum):
    PENDING  = "pending"
    TRAINING = "training"
    READY    = "ready"
    FAILED   = "failed"


class JobStatus(str, enum.Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class ServerGroup(Base):
    """
    A named cluster of servers that share the same business metric.

    Example: "api-cluster-prod" with business metric formula
    sum(rate(orders_total[1m])) — one group, many servers.

    The business metric is defined at the group level because it is
    an application-level signal shared across all servers in the cluster.
    """

    __tablename__ = "server_groups"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Business metric — shared across all servers in this group
    business_metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    business_metric_formula: Mapped[str] = mapped_column(String(1024), nullable=False)
    # Prometheus endpoint for the business metric (may differ from server metrics)
    metrics_host: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics_port: Mapped[int] = mapped_column(nullable=False, default=9090)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    servers: Mapped[list["Server"]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ServerGroup id={self.id} name={self.name!r} servers={len(self.servers or [])}>"


class Server(Base):
    """
    An individual server (node) within a ServerGroup.

    Each server has its own Prometheus endpoint for system metrics
    (CPU, RAM, network) and its own trained model, since per-server
    load patterns may differ even within the same cluster.
    """

    __tablename__ = "servers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    group_id: Mapped[int] = mapped_column(
        ForeignKey("server_groups.id"), nullable=False, index=True
    )
    # Human-readable label for this server within the group
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Prometheus endpoint for this specific server's system metrics
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(nullable=False, default=9090)
    # Optional freeform metadata (role, datacenter, rack, etc.)
    tags: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    group: Mapped["ServerGroup"] = relationship(back_populates="servers")
    configs: Mapped[list["ForecastingConfig"]] = relationship(
        back_populates="server", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Server id={self.id} name={self.name!r} host={self.host}>"


class ForecastingConfig(Base):
    """
    Stores per-server ML configuration (host + business metric formula).

    Each ForecastingConfig belongs to one Server inside a ServerGroup.
    The business metric formula is inherited from the ServerGroup but
    stored here for denormalisation / fast lookup.

    Legacy configs created without a server_id are still supported —
    server_id is nullable to preserve backward compatibility.
    """

    __tablename__ = "forecasting_configs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    # Optional FK to Server (NULL for legacy single-server configs)
    server_id: Mapped[int | None] = mapped_column(
        ForeignKey("servers.id"), nullable=True, index=True
    )
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(nullable=False, default=9090)
    business_metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    business_metric_formula: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    server: Mapped["Server | None"] = relationship(back_populates="configs")
    trained_models: Mapped[list["TrainedModel"]] = relationship(
        back_populates="config", cascade="all, delete-orphan"
    )
    forecasts: Mapped[list["ForecastResult"]] = relationship(
        back_populates="config", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ForecastingConfig id={self.id} name={self.name!r}>"


class TrainedModel(Base):
    """Stores metadata and parameters of trained ML models."""

    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    config_id: Mapped[int] = mapped_column(
        ForeignKey("forecasting_configs.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(nullable=False, default=1)
    algorithm: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[ModelStatus] = mapped_column(
        SAEnum(ModelStatus), default=ModelStatus.PENDING
    )
    # Model parameters / hyperparameters serialized as JSON
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Evaluation metrics on the test split
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Path to the serialized model file on disk (e.g. joblib dump)
    artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # Lag (in minutes) discovered by correlation analysis
    lag_minutes: Mapped[int | None] = mapped_column(nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    config: Mapped["ForecastingConfig"] = relationship(back_populates="trained_models")
    forecasts: Mapped[list["ForecastResult"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<TrainedModel id={self.id} config_id={self.config_id} v{self.version}>"


class ForecastResult(Base):
    """Stores forecast outputs produced for a given business metric value."""

    __tablename__ = "forecast_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    config_id: Mapped[int] = mapped_column(
        ForeignKey("forecasting_configs.id"), nullable=False, index=True
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("trained_models.id"), nullable=False, index=True
    )
    # Input
    business_metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    # Point predictions
    predicted_cpu_percent: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_ram_gb: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_network_mbps: Mapped[float] = mapped_column(Float, nullable=False)
    # 80% prediction interval lower bounds (10th percentile)
    lower_cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    lower_ram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    lower_network_mbps: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 80% prediction interval upper bounds (90th percentile)
    upper_cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_ram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_network_mbps: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Actual values — populated by the accuracy monitor after the fact
    actual_cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_ram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_network_mbps: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Timestamp at which the actual values were fetched
    actuals_fetched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    config: Mapped["ForecastingConfig"] = relationship(back_populates="forecasts")
    model: Mapped["TrainedModel"] = relationship(back_populates="forecasts")

    def __repr__(self) -> str:
        return f"<ForecastResult id={self.id} cpu={self.predicted_cpu_percent:.1f}%>"


class ForecastHorizonResult(Base):
    """
    Stores a multi-step forecast — one row per step in the horizon.

    When an engineer provides a schedule of expected business metric values
    (e.g. every 5 minutes for the next hour), this table holds the full
    time-series response.  Each row links back to the parent ForecastResult
    (the first step) for traceability.
    """

    __tablename__ = "forecast_horizon_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    config_id: Mapped[int] = mapped_column(
        ForeignKey("forecasting_configs.id"), nullable=False, index=True
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("trained_models.id"), nullable=False, index=True
    )
    # Step index within the horizon (0-based)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    # Minutes ahead of the request time that this step represents
    minutes_ahead: Mapped[int] = mapped_column(Integer, nullable=False)
    business_metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    # Point predictions
    predicted_cpu_percent: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_ram_gb: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_network_mbps: Mapped[float] = mapped_column(Float, nullable=False)
    # Prediction intervals
    lower_cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    lower_ram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    lower_network_mbps: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_ram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_network_mbps: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<ForecastHorizonResult config={self.config_id} "
            f"step={self.step} +{self.minutes_ahead}min "
            f"cpu={self.predicted_cpu_percent:.1f}%>"
        )


class ModelEvaluation(Base):
    """
    Stores post-deployment accuracy metrics computed by the accuracy monitor.

    One row per (model, evaluation_run). Evaluations accumulate over time so
    engineers can see how accuracy drifts after a model is deployed.
    """

    __tablename__ = "model_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("trained_models.id"), nullable=False, index=True
    )
    config_id: Mapped[int] = mapped_column(
        ForeignKey("forecasting_configs.id"), nullable=False, index=True
    )
    # Number of forecast-actual pairs used in this evaluation
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    # Per-target metrics (post-deployment, computed against real Prometheus values)
    mae_cpu: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_ram: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_net: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse_cpu: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse_ram: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse_net: Mapped[float | None] = mapped_column(Float, nullable=True)
    mape_overall: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2_cpu: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2_ram: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2_net: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Whether this evaluation triggered a retrain
    triggered_retrain: Mapped[bool] = mapped_column(Boolean, default=False)
    # PSI drift detection results
    psi_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    psi_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model: Mapped["TrainedModel"] = relationship()
    config: Mapped["ForecastingConfig"] = relationship()

    def __repr__(self) -> str:
        return (
            f"<ModelEvaluation id={self.id} model_id={self.model_id} "
            f"n={self.n_samples} r2_cpu={self.r2_cpu}>"
        )


class TrainingJob(Base):
    """
    Tracks an async training run.  One row is created when the engineer
    calls POST /configs/{id}/train/; the background worker updates it as
    the job progresses.

    Lifecycle:  QUEUED → RUNNING → DONE
                                 ↘ FAILED
    """

    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    config_id: Mapped[int] = mapped_column(
        ForeignKey("forecasting_configs.id"), nullable=False, index=True
    )
    # The model produced by this job (set when status → DONE)
    model_id: Mapped[int | None] = mapped_column(
        ForeignKey("trained_models.id"), nullable=True
    )
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus), default=JobStatus.QUEUED, nullable=False
    )
    lookback_days: Mapped[int] = mapped_column(Integer, default=30)
    # Captured traceback if status → FAILED
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    config: Mapped["ForecastingConfig"] = relationship()
    model: Mapped["TrainedModel | None"] = relationship()

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def __repr__(self) -> str:
        return f"<TrainingJob id={self.id} config_id={self.config_id} status={self.status}>"
