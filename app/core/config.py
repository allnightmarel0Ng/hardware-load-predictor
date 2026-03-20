from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://predictor:predictor@localhost:5432/predictor"
    metrics_source_url: str = "http://localhost:9090"
    model_storage_path: str = "./models"
    accuracy_threshold: float = 0.85
    retrain_interval_hours: int = 24
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
