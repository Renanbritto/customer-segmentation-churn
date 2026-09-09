"""Unit tests for RFM feature engineering."""

from datetime import datetime

import pandas as pd
import pytest

from src.features.rfm import RFMEngine


def test_generate_synthetic_transactions():
    """Validates synthetic transactions generator output structure."""
    engine = RFMEngine()
    tx_df, cust_df = engine.generate_synthetic_transactions(n_customers=200, random_state=42)

    assert len(cust_df) == 200
    assert len(tx_df) > 200
    assert set(tx_df.columns) == {"transaction_id", "customer_id", "transaction_date", "amount"}
    assert tx_df["amount"].min() > 0


def test_calculate_rfm_metrics():
    """Validates RFM aggregation and quantile scoring (1-5)."""
    engine = RFMEngine()
    tx_df, _ = engine.generate_synthetic_transactions(n_customers=250, random_state=42)
    rfm = engine.calculate_rfm(tx_df, reference_date=datetime(2026, 9, 1))

    assert len(rfm) == 250
    assert "recency" in rfm.columns
    assert "frequency" in rfm.columns
    assert "monetary" in rfm.columns
    assert "rfm_score" in rfm.columns

    # Quantile scores must be integers between 1 and 5
    assert set(rfm["r_score"].unique()).issubset({1, 2, 3, 4, 5})
    assert set(rfm["f_score"].unique()).issubset({1, 2, 3, 4, 5})
    assert set(rfm["m_score"].unique()).issubset({1, 2, 3, 4, 5})


def test_calculate_rfm_missing_column():
    """Ensures informative ValueError when transactions are missing columns."""
    engine = RFMEngine()
    invalid_df = pd.DataFrame([{"customer_id": "C1", "amount": 100.0}])

    with pytest.raises(ValueError, match="Missing required columns"):
        engine.calculate_rfm(invalid_df)
