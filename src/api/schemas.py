"""Pydantic v2 schemas for Customer Segmentation & Churn API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CustomerRFMInput(BaseModel):
    """Input features representing an individual customer profile."""
    customer_id: Optional[str] = Field(default=None, description="Unique customer identifier", examples=["CUST-00123"])
    recency: int = Field(..., ge=0, description="Inactivity days since most recent transaction", examples=[18])
    frequency: int = Field(..., ge=1, description="Total transaction or order count", examples=[9])
    monetary: float = Field(..., gt=0.0, description="Total historical spend (R$)", examples=[5530.0])


class CustomerSegmentationResponse(BaseModel):
    """Segment assignment, persona identification and churn health score."""
    customer_id: Optional[str]
    cluster_id: int
    persona_key: str
    persona_name: str
    recency: int
    frequency: int
    monetary: float
    avg_ticket: float
    churn_probability: float
    retention_probability: float
    risk_tier: str
    primary_risk_driver: str
    retention_projection: Dict[str, float]
    action_playbook: str
    distance_to_centroid: float


class BatchSegmentationRequest(BaseModel):
    """Payload for batch customer scoring."""
    customers: List[CustomerRFMInput] = Field(..., min_length=1, max_length=1000)


class BatchSegmentationResponse(BaseModel):
    """Summary of batch scoring run."""
    total_customers: int
    results: List[CustomerSegmentationResponse]
    persona_distribution: Dict[str, int]


class ElbowMetricsResponse(BaseModel):
    """Clustering evaluation curve across k values."""
    metrics: List[Dict[str, Any]]
    optimal_k: int
    recommended_rationale: str


class PersonasCatalogResponse(BaseModel):
    """Catalog of business personas and operational playbooks."""
    personas: List[Dict[str, Any]]
    cohort_retention_benchmark: List[Dict[str, Any]]
