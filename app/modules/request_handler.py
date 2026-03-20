"""
Module 7 — Request Handler
FastAPI routers that expose the system's REST API.
This module wires together all other modules in response to HTTP requests.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules import (
    config_manager,
    data_collector,
    correlation_analyzer,
    model_trainer,
    forecasting_engine,
)
from app.schemas.schemas import (
    ForecastingConfigCreate,
    ForecastingConfigRead,
    ForecastingConfigUpdate,
    ForecastRequest,
    ForecastResponse,
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

    bundle = data_collector.fetch_historical_data(
        host=config.host,
        port=config.port,
        business_formula=config.business_metric_formula,
        lookback_days=body.lookback_days,
    )

    report = correlation_analyzer.analyze(bundle)
    if not report.any_significant:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No significant correlations found between the business metric "
                "and system metrics. Check your formula or try more data."
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
