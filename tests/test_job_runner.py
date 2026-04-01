"""
Tests for job_runner.py — async training job submission and lifecycle.

Strategy: patch _run_training_job to avoid actually running training in
thread-pool threads during tests, then test the job lifecycle state machine.
"""
import time
import threading
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from app.models.db_models import JobStatus, TrainingJob
from app.modules import job_runner
from app.modules.config_manager import create_config
from app.schemas.schemas import ForecastingConfigCreate


# ── fixtures ──────────────────────────────────────────────────────────────────

def _cfg(name: str) -> ForecastingConfigCreate:
    return ForecastingConfigCreate(
        name=name, host="prometheus.internal", port=9090,
        business_metric_name="orders",
        business_metric_formula="sum(rate(orders_total[1m]))",
    )


# ── submit_training_job ───────────────────────────────────────────────────────

class TestSubmitTrainingJob:
    def test_creates_queued_job_record(self, db):
        cfg = create_config(db, _cfg("job-create"))
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            job = job_runner.submit_training_job(cfg.id, lookback_days=14)

        assert job.id is not None
        assert job.config_id == cfg.id
        assert job.status == JobStatus.QUEUED
        assert job.lookback_days == 14
        assert job.started_at  is None
        assert job.finished_at is None
        assert job.model_id    is None
        assert job.error_message is None

    def test_submits_to_executor(self, db):
        cfg = create_config(db, _cfg("job-submit"))
        submitted = []
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = lambda fn, job_id: submitted.append((fn, job_id))
            job = job_runner.submit_training_job(cfg.id, lookback_days=7)

        assert len(submitted) == 1
        fn, job_id = submitted[0]
        assert fn == job_runner._run_training_job
        assert job_id == job.id

    def test_raises_if_executor_not_started(self, db):
        cfg = create_config(db, _cfg("job-no-exec"))
        with patch.object(job_runner, "_executor", None):
            with pytest.raises(RuntimeError, match="executor is not running"):
                job_runner.submit_training_job(cfg.id, lookback_days=7)

    def test_multiple_jobs_get_distinct_ids(self, db):
        cfg = create_config(db, _cfg("job-multi"))
        ids = []
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            for _ in range(3):
                j = job_runner.submit_training_job(cfg.id, lookback_days=7)
                ids.append(j.id)
        assert len(set(ids)) == 3


# ── get_job / list_jobs ───────────────────────────────────────────────────────

class TestGetAndListJobs:
    def test_get_job_returns_correct_record(self, db):
        cfg = create_config(db, _cfg("job-get"))
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            submitted = job_runner.submit_training_job(cfg.id, lookback_days=14)

        fetched = job_runner.get_job(submitted.id)
        assert fetched is not None
        assert fetched.id == submitted.id
        assert fetched.config_id == cfg.id

    def test_get_job_returns_none_for_missing(self, db):
        result = job_runner.get_job(999_999)
        assert result is None

    def test_list_jobs_returns_for_config(self, db):
        cfg = create_config(db, _cfg("job-list"))
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            for _ in range(3):
                job_runner.submit_training_job(cfg.id, lookback_days=7)

        jobs = job_runner.list_jobs(cfg.id)
        assert len(jobs) == 3
        assert all(j.config_id == cfg.id for j in jobs)

    def test_list_jobs_respects_limit(self, db):
        cfg = create_config(db, _cfg("job-limit"))
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            for _ in range(5):
                job_runner.submit_training_job(cfg.id, lookback_days=7)

        jobs = job_runner.list_jobs(cfg.id, limit=2)
        assert len(jobs) <= 2

    def test_list_jobs_ordered_newest_first(self, db):
        cfg = create_config(db, _cfg("job-order"))
        created_ids = []
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            for _ in range(3):
                j = job_runner.submit_training_job(cfg.id, lookback_days=7)
                created_ids.append(j.id)

        jobs = job_runner.list_jobs(cfg.id)
        listed_ids = [j.id for j in jobs]
        assert listed_ids == sorted(created_ids, reverse=True)


# ── _run_training_job state machine ──────────────────────────────────────────

class TestRunTrainingJobStateMachine:
    """
    Test the worker function's state transitions without actually training.
    We directly call _run_training_job with mocked pipeline functions.
    """

    def _create_queued_job(self, db, name: str) -> tuple:
        cfg = create_config(db, _cfg(name))
        job = TrainingJob(
            config_id=cfg.id,
            lookback_days=7,
            status=JobStatus.QUEUED,
        )
        db.add(job); db.commit(); db.refresh(job)
        return cfg, job

    def test_successful_run_sets_done_status(self, db):
        from app.modules.data_collector import MetricsBundle
        from app.modules.correlation_analyzer import CorrelationReport, CorrelationResult
        cfg, job = self._create_queued_job(db, "run-done")

        fake_model = MagicMock()
        fake_model.id = 42
        fake_model.algorithm = "gradient_boosting"
        fake_model.metrics = {"r2_cpu": 0.91}

        fake_bundle = MagicMock(spec=MetricsBundle)
        fake_report = MagicMock(spec=CorrelationReport)
        fake_report.any_significant = True

        with patch("app.modules.job_runner.fetch_historical_data", return_value=fake_bundle), \
             patch("app.modules.job_runner.analyze",               return_value=fake_report), \
             patch("app.modules.job_runner.train_model",           return_value=fake_model):
            job_runner._run_training_job(job.id)

        db.expire_all()
        updated = db.get(TrainingJob, job.id)
        assert updated.status     == JobStatus.DONE
        assert updated.model_id   == 42
        assert updated.started_at  is not None
        assert updated.finished_at is not None

    def test_failed_run_sets_failed_status(self, db):
        cfg, job = self._create_queued_job(db, "run-fail")

        with patch("app.modules.job_runner.fetch_historical_data",
                   side_effect=RuntimeError("Prometheus unavailable")):
            job_runner._run_training_job(job.id)

        db.expire_all()
        updated = db.get(TrainingJob, job.id)
        assert updated.status == JobStatus.FAILED
        assert updated.error_message is not None
        assert "Prometheus unavailable" in updated.error_message
        assert updated.finished_at is not None

    def test_no_significant_correlation_fails_job(self, db):
        from app.modules.data_collector import MetricsBundle
        from app.modules.correlation_analyzer import CorrelationReport
        cfg, job = self._create_queued_job(db, "run-no-corr")

        fake_bundle = MagicMock(spec=MetricsBundle)
        fake_report = MagicMock(spec=CorrelationReport)
        fake_report.any_significant = False

        with patch("app.modules.job_runner.fetch_historical_data", return_value=fake_bundle), \
             patch("app.modules.job_runner.analyze",               return_value=fake_report):
            job_runner._run_training_job(job.id)

        db.expire_all()
        updated = db.get(TrainingJob, job.id)
        assert updated.status == JobStatus.FAILED
        assert "correlations" in (updated.error_message or "").lower()

    def test_missing_job_id_is_handled_gracefully(self, db):
        # Should not raise
        job_runner._run_training_job(999_999)

    def test_job_transitions_through_running_state(self, db):
        """Verify started_at is set before training completes."""
        from app.modules.data_collector import MetricsBundle
        from app.modules.correlation_analyzer import CorrelationReport
        cfg, job = self._create_queued_job(db, "run-running")

        started_at_during = {}

        def slow_analyze(bundle):
            # Read the job status mid-execution
            db2 = job_runner.SessionLocal()
            try:
                j = db2.get(TrainingJob, job.id)
                started_at_during["status"]     = j.status
                started_at_during["started_at"] = j.started_at
            finally:
                db2.close()
            report = MagicMock(spec=CorrelationReport)
            report.any_significant = True
            return report

        fake_model = MagicMock()
        fake_model.id = 99
        fake_model.algorithm = "gradient_boosting"
        fake_model.metrics = {"r2_cpu": 0.9}

        with patch("app.modules.job_runner.fetch_historical_data",
                   return_value=MagicMock(spec=MetricsBundle)), \
             patch("app.modules.job_runner.analyze", side_effect=slow_analyze), \
             patch("app.modules.job_runner.train_model", return_value=fake_model):
            job_runner._run_training_job(job.id)

        assert started_at_during.get("status") == JobStatus.RUNNING
        assert started_at_during.get("started_at") is not None

    def test_duration_seconds_computed_correctly(self, db):
        cfg, job = self._create_queued_job(db, "run-duration")

        fake_model = MagicMock()
        fake_model.id = 1; fake_model.algorithm = "ridge"; fake_model.metrics = {}

        from app.modules.data_collector import MetricsBundle
        from app.modules.correlation_analyzer import CorrelationReport
        fake_report = MagicMock(spec=CorrelationReport); fake_report.any_significant = True

        with patch("app.modules.job_runner.fetch_historical_data",
                   return_value=MagicMock(spec=MetricsBundle)), \
             patch("app.modules.job_runner.analyze", return_value=fake_report), \
             patch("app.modules.job_runner.train_model", return_value=fake_model):
            job_runner._run_training_job(job.id)

        db.expire_all()
        updated = db.get(TrainingJob, job.id)
        assert updated.duration_seconds is not None
        assert updated.duration_seconds >= 0.0


# ── Executor lifecycle ────────────────────────────────────────────────────────

class TestExecutorLifecycle:
    def test_start_creates_executor(self):
        original = job_runner._executor
        try:
            job_runner._executor = None
            job_runner.start_executor()
            assert job_runner._executor is not None
        finally:
            job_runner.stop_executor()
            job_runner._executor = original

    def test_stop_cleans_up_executor(self):
        original = job_runner._executor
        try:
            job_runner._executor = None
            job_runner.start_executor()
            job_runner.stop_executor()
            assert job_runner._executor is None
        finally:
            job_runner._executor = original

    def test_double_start_is_idempotent(self):
        original = job_runner._executor
        try:
            job_runner._executor = None
            job_runner.start_executor()
            exec1 = job_runner._executor
            job_runner.start_executor()   # second call — should be a no-op
            assert job_runner._executor is exec1
        finally:
            job_runner.stop_executor()
            job_runner._executor = original


# ── API endpoint tests ────────────────────────────────────────────────────────

class TestJobAPIEndpoints:
    def _submit_job(self, client, name: str) -> dict:
        cfg = client.post("/configs/", json={
            "name": name, "host": "h", "port": 9090,
            "business_metric_name": "orders",
            "business_metric_formula": "orders_total",
        }).json()
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            resp = client.post(
                f"/configs/{cfg['id']}/train/",
                json={"lookback_days": 7},
            )
        return resp.json()

    def test_train_returns_202_and_job_id(self, client):
        cfg_r = client.post("/configs/", json={
            "name": "api-job-202", "host": "h", "port": 9090,
            "business_metric_name": "m", "business_metric_formula": "m_total",
        })
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            r = client.post(f"/configs/{cfg_r.json()['id']}/train/",
                            json={"lookback_days": 7})
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "queued"

    def test_get_job_returns_queued_status(self, client):
        job = self._submit_job(client, "api-job-get")
        r = client.get(f"/jobs/{job['job_id']}")
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        assert r.json()["id"] == job["job_id"]

    def test_get_missing_job_returns_404(self, client):
        r = client.get("/jobs/999999")
        assert r.status_code == 404

    def test_list_jobs_for_config(self, client):
        cfg_r = client.post("/configs/", json={
            "name": "api-job-list", "host": "h", "port": 9090,
            "business_metric_name": "m", "business_metric_formula": "m_total",
        })
        config_id = cfg_r.json()["id"]
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            for _ in range(3):
                client.post(f"/configs/{config_id}/train/", json={"lookback_days": 7})

        r = client.get(f"/configs/{config_id}/train/jobs")
        assert r.status_code == 200
        assert len(r.json()) == 3

    def test_train_404_for_missing_config(self, client):
        with patch.object(job_runner, "_executor") as mock_exec:
            mock_exec.submit = MagicMock()
            r = client.post("/configs/999999/train/", json={"lookback_days": 7})
        assert r.status_code == 404
