"""Churn Propensity and Customer Health Scoring module.

Computes calibrated churn probabilities, assigns risk tiers, decomposes
key risk drivers, and projects cohort retention decay curves.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ChurnRiskProfile:
    """Customer-level churn risk assessment and action directive."""
    churn_probability: float
    retention_probability: float
    risk_tier: str
    primary_risk_driver: str
    retention_projection: Dict[str, float]
    action_plan: str


class ChurnRiskEstimator:
    """Calibrated churn hazard estimator based on RFM decay dynamics."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def estimate_churn(
        recency_days: float,
        frequency: float,
        monetary: float,
        persona_key: Optional[str] = None,
    ) -> ChurnRiskProfile:
        """Estimates churn probability using calibrated logistic decay.

        Args:
            recency_days: Inactivity days since last transaction.
            frequency: Lifetime order frequency.
            monetary: Total historical revenue generated.
            persona_key: Optional persona identifier.
        """
        if recency_days < 0:
            raise ValueError("Recency days cannot be negative.")
        if frequency < 0 or monetary < 0:
            raise ValueError("Frequency and monetary cannot be negative.")

        # Baseline log-odds factors
        # Recency is the strongest positive driver of churn hazard
        # Frequency and Monetary provide buffers against churn
        z = -2.80 + (0.042 * recency_days) - (0.12 * frequency) - (0.00015 * monetary)

        # Persona prior adjustment if available
        if persona_key == "vip":
            z -= 0.60
        elif persona_key == "potential":
            z -= 0.20
        elif persona_key == "at_risk":
            z += 0.80
        elif persona_key == "hibernating":
            z += 1.40

        prob = float(1.0 / (1.0 + np.exp(-z)))
        prob = max(0.005, min(0.995, prob))
        retention_prob = 1.0 - prob

        # Risk tier assignment
        if prob < 0.15:
            risk_tier = "Baixo Risco"
            primary_driver = "Engajamento continuo e compras recentes"
            action = "Manter rotina de comunicacao e explorar oportunidades de cross-sell/upsell."
        elif prob < 0.40:
            risk_tier = "Risco Moderado"
            primary_driver = "Intervalo de compra acima da media esperada"
            action = "Disparar campanha de nutricao com beneficios de recompra e desconto progressivo."
        elif prob < 0.70:
            risk_tier = "Alto Risco"
            primary_driver = "Recencia critica e queda acentuada na frequencia"
            action = "Contato direto de Customer Success / Consultor comercial com oferta de retencao."
        else:
            risk_tier = "Risco Critico"
            primary_driver = "Inatividade severa e desconexao da base ativa"
            action = "Campanha agressiva de win-back ou reativacao por canais de baixo custo."

        # Cohort retention projection across M0 to M6
        # Exponential survival model: S(t) = exp(-hazard * t)
        hazard = -np.log(max(0.01, retention_prob)) / 6.0
        retention_projection = {
            f"M{t}": round(float(np.exp(-hazard * t) * 100.0), 1)
            for t in range(7)
        }

        return ChurnRiskProfile(
            churn_probability=round(prob, 4),
            retention_probability=round(retention_prob, 4),
            risk_tier=risk_tier,
            primary_risk_driver=primary_driver,
            retention_projection=retention_projection,
            action_plan=action,
        )
