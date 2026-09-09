"""RFM (Recency, Frequency, Monetary) Feature Engineering module.

Computes transactional customer metrics, quantile-based scoring (1-5),
and provides synthetic transaction generation adhering to B2B/B2C archetypes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RFMRecord:
    """Individual customer RFM metrics and score."""
    customer_id: str
    recency: int
    frequency: int
    monetary: float
    avg_ticket: float
    r_score: int
    f_score: int
    m_score: int
    rfm_score: str


class RFMEngine:
    """Engine for RFM feature extraction and quantile scoring."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_synthetic_transactions(
        n_customers: int = 3000,
        reference_date: Optional[datetime] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates realistic transaction logs and expected ground-truth customer profiles.

        Args:
            n_customers: Total unique customers to generate (default 3000).
            reference_date: Cutoff evaluation date (defaults to 2026-09-01).
            random_state: Reproducibility seed.

        Returns:
            Tuple of (transactions_df, customer_ground_truth_df).
        """
        if n_customers < 100:
            raise ValueError("n_customers must be at least 100.")

        if reference_date is None:
            ref_date = datetime(2026, 9, 1)
        else:
            ref_date = reference_date

        rng = np.random.default_rng(random_state)

        # Persona distributions: VIP (14%), Potential (29%), At-Risk (18%), Hibernating (39%)
        n_vip = int(n_customers * 0.14)
        n_potential = int(n_customers * 0.29)
        n_at_risk = int(n_customers * 0.18)
        n_hibernating = n_customers - (n_vip + n_potential + n_at_risk)

        personas_config = [
            ("VIP", n_vip, (1, 8), (16, 26), (400.0, 650.0)),
            ("Potential", n_potential, (10, 28), (7, 14), (380.0, 580.0)),
            ("At_Risk", n_at_risk, (60, 105), (3, 7), (900.0, 1500.0)),
            ("Hibernating", n_hibernating, (105, 220), (1, 3), (300.0, 600.0)),
        ]

        transactions_list = []
        customer_records = []
        cust_counter = 1
        tx_counter = 1

        for persona_name, count, rec_range, freq_range, ticket_range in personas_config:
            for _ in range(count):
                cid = f"CUST-{cust_counter:05d}"
                cust_counter += 1

                recency_days = int(rng.integers(rec_range[0], rec_range[1] + 1))
                freq = int(rng.integers(freq_range[0], freq_range[1] + 1))

                last_purchase_date = ref_date - timedelta(days=recency_days)

                total_spend = 0.0
                # Generate transaction timestamps spread backwards across the customer lifespan
                for f_idx in range(freq):
                    ticket = float(rng.uniform(ticket_range[0], ticket_range[1]))
                    total_spend += ticket

                    if f_idx == 0:
                        tx_date = last_purchase_date
                    else:
                        days_prior = int(rng.integers(15, 300))
                        tx_date = last_purchase_date - timedelta(days=days_prior)

                    transactions_list.append({
                        "transaction_id": f"TX-{tx_counter:07d}",
                        "customer_id": cid,
                        "transaction_date": tx_date.strftime("%Y-%m-%d"),
                        "amount": round(ticket, 2),
                    })
                    tx_counter += 1

                customer_records.append({
                    "customer_id": cid,
                    "persona_ground_truth": persona_name,
                    "target_recency": recency_days,
                    "target_frequency": freq,
                    "target_monetary": round(total_spend, 2),
                })

        tx_df = pd.DataFrame(transactions_list)
        cust_df = pd.DataFrame(customer_records)

        return tx_df, cust_df

    def calculate_rfm(
        self,
        transactions_df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Aggregates transactional log into customer RFM dataframe with 1-5 quantile scores.

        Args:
            transactions_df: DataFrame with ['customer_id', 'transaction_date', 'amount'].
            reference_date: Reference analysis date. Defaults to max(transaction_date) + 1 day.
        """
        required_cols = {"customer_id", "transaction_date", "amount"}
        if not required_cols.issubset(transactions_df.columns):
            missing = required_cols - set(transactions_df.columns)
            raise ValueError(f"Missing required columns in transactions DataFrame: {missing}")

        df = transactions_df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        if reference_date is None:
            ref_date = df["transaction_date"].max() + timedelta(days=1)
        else:
            ref_date = pd.to_datetime(reference_date)

        # Aggregate metrics
        rfm = df.groupby("customer_id").agg(
            last_date=("transaction_date", "max"),
            frequency=("transaction_id" if "transaction_id" in df.columns else "amount", "count"),
            monetary=("amount", "sum"),
        ).reset_index()

        rfm["recency"] = (ref_date - rfm["last_date"]).dt.days
        rfm["monetary"] = rfm["monetary"].round(2)
        rfm["avg_ticket"] = (rfm["monetary"] / rfm["frequency"]).round(2)

        # Quantile scoring 1 to 5
        # Recency: lower is better -> highest score 5 for lowest recency
        rfm["r_score"] = pd.qcut(
            rfm["recency"].rank(method="first"),
            q=5,
            labels=[5, 4, 3, 2, 1],
        ).astype(int)

        # Frequency: higher is better
        rfm["f_score"] = pd.qcut(
            rfm["frequency"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

        # Monetary: higher is better
        rfm["m_score"] = pd.qcut(
            rfm["monetary"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

        rfm["rfm_score"] = (
            rfm["r_score"].astype(str)
            + "-"
            + rfm["f_score"].astype(str)
            + "-"
            + rfm["m_score"].astype(str)
        )

        return rfm[[
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "avg_ticket",
            "r_score",
            "f_score",
            "m_score",
            "rfm_score",
        ]]
