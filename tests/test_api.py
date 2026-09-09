"""Integration tests for FastAPI endpoints."""

from fastapi import status


def test_health_endpoint(api_client):
    """Checks /health endpoint."""
    resp = api_client.get("/health")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["status"] == "healthy"


def test_personas_catalog_endpoint(api_client):
    """Checks /personas/catalog endpoint."""
    resp = api_client.get("/personas/catalog")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert len(data["personas"]) == 4
    assert len(data["cohort_retention_benchmark"]) == 7


def test_elbow_metrics_endpoint(api_client):
    """Checks /metrics/elbow endpoint."""
    resp = api_client.get("/metrics/elbow")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["optimal_k"] == 4
    assert len(data["metrics"]) > 0


def test_single_prediction_endpoint(api_client):
    """Checks /segmentation/predict endpoint."""
    payload = {
        "customer_id": "CUST-TEST-01",
        "recency": 4,
        "frequency": 22,
        "monetary": 11500.0,
    }
    resp = api_client.post("/segmentation/predict", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["customer_id"] == "CUST-TEST-01"
    assert "persona_name" in data
    assert data["churn_probability"] < 0.20
    assert "action_playbook" in data


def test_batch_prediction_endpoint(api_client):
    """Checks /segmentation/batch endpoint."""
    payload = {
        "customers": [
            {"customer_id": "C1", "recency": 3, "frequency": 20, "monetary": 10000.0},
            {"customer_id": "C2", "recency": 150, "frequency": 2, "monetary": 800.0},
        ]
    }
    resp = api_client.post("/segmentation/batch", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["total_customers"] == 2
    assert len(data["results"]) == 2
    assert "persona_distribution" in data
