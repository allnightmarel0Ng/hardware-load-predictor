from datetime import datetime
from sqlalchemy import String, Float, DateTime, JSON, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.database import Base


class ModelStatus(str, enum.Enum):
    PENDING = "pending"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"


class ForecastingConfig(Base):
    """Stores engineer-defined configurations (host + business metric formula)."""

    __tablename__ = "forecasting_configs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(nullable=False, default=9090)
    business_metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    business_metric_formula: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

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
    # Outputs
    predicted_cpu_percent: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_ram_gb: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_network_mbps: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    config: Mapped["ForecastingConfig"] = relationship(back_populates="forecasts")
    model: Mapped["TrainedModel"] = relationship(back_populates="forecasts")

    def __repr__(self) -> str:
        return f"<ForecastResult id={self.id} cpu={self.predicted_cpu_percent:.1f}%>"
