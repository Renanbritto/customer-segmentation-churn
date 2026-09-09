"""FastAPI route endpoints for Customer Segmentation & Churn API."""

from collections import Counter
from typing import Dict

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    BatchSegmentationRequest,
    BatchSegmentationResponse,
    CustomerRFMInput,
    CustomerSegmentationResponse,
    ElbowMetricsResponse,
    PersonasCatalogResponse,
)
from src.config import COHORT_BENCHMARK, ELBOW_REFERENCE_DATA, PERSONAS_METADATA
from src.models.churn import ChurnRiskEstimator
from src.models.kmeans import KMeansSegmentation

router = APIRouter()

# Global fitted segmentation model instance with reference centroids
_KMEANS_MODEL = KMeansSegmentation(n_clusters=4)
# Initialize with reference archetypes to ensure immediate readiness
_ref_profiles = pd.DataFrame([
    {"recency": 4, "frequency": 22, "monetary": 11000.0},
    {"recency": 18, "frequency": 10, "monetary": 5600.0},
    {"recency": 78, "frequency": 5, "monetary": 7200.0},
    {"recency": 145, "frequency": 2, "monetary": 1150.0},
])
_KMEANS_MODEL.fit(_ref_profiles)


@router.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health_check() -> Dict[str, str]:
    """Health check for deployment monitoring."""
    return {"status": "healthy", "service": "customer-segmentation-churn"}


@router.get(
    "/personas/catalog",
    response_model=PersonasCatalogResponse,
    status_code=status.HTTP_200_OK,
    tags=["Segmentation"],
    summary="Retrieve personas catalog and cohort retention benchmark",
)
def get_personas_catalog() -> PersonasCatalogResponse:
    """Returns business personas metadata and historical cohort retention rates."""
    personas_list = list(PERSONAS_METADATA.values())
    return PersonasCatalogResponse(
        personas=personas_list,
        cohort_retention_benchmark=COHORT_BENCHMARK,
    )


@router.get(
    "/metrics/elbow",
    response_model=ElbowMetricsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Evaluation"],
    summary="Get Elbow Inertia and Silhouette Score curves",
)
def get_elbow_metrics() -> ElbowMetricsResponse:
    """Returns clustering quality scores across k in [2, 8] demonstrating optimal k=4."""
    return ElbowMetricsResponse(
        metrics=ELBOW_REFERENCE_DATA,
        optimal_k=4,
        recommended_rationale=(
            "O valor k=4 maximiza o Silhouette Score (0.68) e representa o ponto de inflexao "
            "da Inercia (Metodo do Cotovelo), isolando com precisao clientes VIPs e contas em risco."
        ),
    )


@router.post(
    "/segmentation/predict",
    response_model=CustomerSegmentationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Score an individual customer profile",
)
def predict_customer_segment(payload: CustomerRFMInput) -> CustomerSegmentationResponse:
    """Classifies a customer into a K-Means cluster and evaluates churn risk."""
    try:
        clust_pred = _KMEANS_MODEL.predict_single(
            recency=payload.recency,
            frequency=payload.frequency,
            monetary=payload.monetary,
        )

        churn_profile = ChurnRiskEstimator.estimate_churn(
            recency_days=float(payload.recency),
            frequency=float(payload.frequency),
            monetary=float(payload.monetary),
            persona_key=clust_pred.persona_key,
        )

        avg_ticket = round(payload.monetary / payload.frequency, 2)

        return CustomerSegmentationResponse(
            customer_id=payload.customer_id,
            cluster_id=clust_pred.cluster_id,
            persona_key=clust_pred.persona_key,
            persona_name=clust_pred.persona_name,
            recency=payload.recency,
            frequency=payload.frequency,
            monetary=payload.monetary,
            avg_ticket=avg_ticket,
            churn_probability=churn_profile.churn_probability,
            retention_probability=churn_profile.retention_probability,
            risk_tier=churn_profile.risk_tier,
            primary_risk_driver=churn_profile.primary_risk_driver,
            retention_projection=churn_profile.retention_projection,
            action_playbook=clust_pred.action_playbook,
            distance_to_centroid=clust_pred.distance_to_centroid,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/segmentation/batch",
    response_model=BatchSegmentationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Batch classify multiple customer profiles",
)
def predict_batch_segments(payload: BatchSegmentationRequest) -> BatchSegmentationResponse:
    """Evaluates a batch of customer profiles."""
    results = [predict_customer_segment(c) for c in payload.customers]
    dist = dict(Counter(r.persona_name for r in results))

    return BatchSegmentationResponse(
        total_customers=len(results),
        results=results,
        persona_distribution=dist,
    )
