"""Unit tests for Churn Propensity Scoring."""

import pytest

from src.models.churn import ChurnRiskEstimator


def test_churn_estimation_vip():
    """Validates that a recent high-frequency customer has very low churn risk."""
    res = ChurnRiskEstimator.estimate_churn(
        recency_days=3.0,
        frequency=20.0,
        monetary=10000.0,
        persona_key="vip",
    )

    assert res.churn_probability < 0.15
    assert res.risk_tier == "Baixo Risco"
    assert res.retention_probability > 0.85
    assert "M0" in res.retention_projection
    assert res.retention_projection["M0"] == 100.0


def test_churn_estimation_inactivity():
    """Validates that long inactivity yields severe churn hazard."""
    res = ChurnRiskEstimator.estimate_churn(
        recency_days=160.0,
        frequency=2.0,
        monetary=1200.0,
        persona_key="hibernating",
    )

    assert res.churn_probability > 0.70
    assert res.risk_tier in ["Alto Risco", "Risco Critico"]


def test_churn_negative_inputs():
    """Ensures errors on negative recency or frequency."""
    with pytest.raises(ValueError, match="Recency days cannot be negative"):
        ChurnRiskEstimator.estimate_churn(recency_days=-5.0, frequency=10.0, monetary=1000.0)
