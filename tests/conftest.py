"""Shared test fixtures for customer segmentation and churn test suite."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.features.rfm import RFMEngine


@pytest.fixture(scope="session")
def api_client():
    """Provides a shared FastAPI TestClient instance."""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def sample_rfm_data():
    """Generates synthetic RFM customer dataframe for unit testing."""
    engine = RFMEngine()
    tx_df, _ = engine.generate_synthetic_transactions(n_customers=500, random_state=42)
    return engine.calculate_rfm(tx_df)
