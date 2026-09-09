"""FastAPI application entrypoint for Customer Segmentation & Churn API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="Customer Segmentation & Churn Prevention API",
    description=(
        "Enterprise Machine Learning API for RFM feature engineering, "
        "K-Means clustering segmentation, persona profiling, and churn risk scoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(router)
