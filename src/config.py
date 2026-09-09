"""Global configuration, domain constants and benchmark datasets.

Defines directory paths, persona profiles, cohort retention benchmarks,
and default hyperparameters for RFM and K-Means segmentation.
"""

from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

RAW_TRANSACTIONS_PATH = DATA_RAW_DIR / "customer_transactions.csv"
PROCESSED_RFM_PATH = DATA_PROCESSED_DIR / "rfm_features.parquet"

# Clustering Defaults
DEFAULT_N_CLUSTERS = 4
RANDOM_STATE = 42

# Personas Definitions & Business Playbooks
PERSONAS_METADATA: Dict[str, Dict[str, Any]] = {
    "vip": {
        "id": 0,
        "name": "Champions / Clientes VIPs",
        "description": "Clientes de altissimo valor, frequencia elevada e compras muito recentes.",
        "avg_recency_days": 4,
        "avg_frequency": 21.4,
        "avg_monetary": 10330.0,
        "base_churn_rate": 0.032,
        "action_playbook": "Atendimento VIP Concierge, conselho de produto, upsell exclusivo e SLAs prioritarios.",
    },
    "potential": {
        "id": 1,
        "name": "Potenciais Leais",
        "description": "Compradores regulares com bom ticket medio e potencial de expansao de conta.",
        "avg_recency_days": 18,
        "avg_frequency": 9.5,
        "avg_monetary": 5530.0,
        "base_churn_rate": 0.125,
        "action_playbook": "Oferta de plano anual com desconto, programas de fidelidade e demonstracao de novas features.",
    },
    "at_risk": {
        "id": 2,
        "name": "Em Risco Alto (Ticket Alto)",
        "description": "Historico de alto investimento, mas com periodo excessivo de inatividade (recencia critica).",
        "avg_recency_days": 78,
        "avg_frequency": 4.6,
        "avg_monetary": 6880.0,
        "base_churn_rate": 0.684,
        "action_playbook": "Contato executivo urgente de Customer Success, diagnostico de insatisfacao e oferta de retencao.",
    },
    "hibernating": {
        "id": 3,
        "name": "Hibernando / Baixo Engajamento",
        "description": "Baixo volume de transacoes, ticket reduzido e longo tempo sem interacao.",
        "avg_recency_days": 142,
        "avg_frequency": 1.8,
        "avg_monetary": 1190.0,
        "base_churn_rate": 0.890,
        "action_playbook": "Automacao de e-mail marketing com cupons agressivos de reativacao ou reducao de CAC.",
    },
}

# Cohort Retention Matrix (M0 a M6)
COHORT_BENCHMARK: List[Dict[str, Any]] = [
    {"month": "M0", "month_label": "M0 (Inicio)", "vip": 100.0, "potential": 100.0, "at_risk": 100.0, "hibernating": 100.0},
    {"month": "M1", "month_label": "M1", "vip": 98.0, "potential": 92.0, "at_risk": 65.0, "hibernating": 42.0},
    {"month": "M2", "month_label": "M2", "vip": 97.0, "potential": 88.0, "at_risk": 48.0, "hibernating": 28.0},
    {"month": "M3", "month_label": "M3", "vip": 96.0, "potential": 84.0, "at_risk": 36.0, "hibernating": 19.0},
    {"month": "M4", "month_label": "M4", "vip": 95.0, "potential": 81.0, "at_risk": 28.0, "hibernating": 14.0},
    {"month": "M5", "month_label": "M5", "vip": 94.0, "potential": 78.0, "at_risk": 22.0, "hibernating": 11.0},
    {"month": "M6", "month_label": "M6", "vip": 94.0, "potential": 76.0, "at_risk": 18.0, "hibernating": 9.0},
]

# Elbow Method & Silhouette Reference Curve
ELBOW_REFERENCE_DATA: List[Dict[str, Any]] = [
    {"k": 2, "inertia": 18500.0, "silhouette": 0.42},
    {"k": 3, "inertia": 11200.0, "silhouette": 0.51},
    {"k": 4, "inertia": 5400.0, "silhouette": 0.68},  # Optimal K
    {"k": 5, "inertia": 4300.0, "silhouette": 0.59},
    {"k": 6, "inertia": 3600.0, "silhouette": 0.54},
    {"k": 7, "inertia": 3100.0, "silhouette": 0.49},
    {"k": 8, "inertia": 2700.0, "silhouette": 0.45},
]
