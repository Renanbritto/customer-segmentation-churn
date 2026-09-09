"""K-Means Clustering module for customer RFM segmentation.

Implements log-standardization scaling pipeline, inertia / silhouette evaluation,
and deterministic persona mapping with business action playbooks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_N_CLUSTERS, PERSONAS_METADATA, RANDOM_STATE


@dataclass(frozen=True)
class ClusteringEvaluation:
    """Evaluation metrics for a given K configuration."""
    k: int
    inertia: float
    silhouette: float
    davies_bouldin: float


@dataclass(frozen=True)
class CustomerClusterPrediction:
    """Inference output for a single customer."""
    cluster_id: int
    persona_key: str
    persona_name: str
    action_playbook: str
    base_churn_rate: float
    distance_to_centroid: float


class KMeansSegmentation:
    """K-Means clustering engine for RFM data."""

    def __init__(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[KMeans] = None
        self.cluster_to_persona_map: Dict[int, str] = {}
        self.fitted_ = False

    def _preprocess_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Applies log1p transformation followed by standard scaling."""
        X_log = np.log1p(np.maximum(X, 0.0))
        if fit:
            return self.scaler.fit_transform(X_log)
        return self.scaler.transform(X_log)

    def fit(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Fits K-Means on RFM features and attaches cluster/persona metadata.

        Args:
            rfm_df: DataFrame containing ['recency', 'frequency', 'monetary'].

        Returns:
            DataFrame with additional columns:
            ['cluster_id', 'persona_key', 'persona_name', 'action_playbook'].
        """
        for col in ["recency", "frequency", "monetary"]:
            if col not in rfm_df.columns:
                raise ValueError(f"Missing required feature column: {col}")

        df = rfm_df.copy()
        raw_features = df[["recency", "frequency", "monetary"]].values.astype(float)
        scaled_features = self._preprocess_features(raw_features, fit=True)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.model.fit_predict(scaled_features)
        df["cluster_id"] = labels

        # Determine persona mapping dynamically based on raw centroid statistics
        self._build_persona_mapping(df)

        df["persona_key"] = df["cluster_id"].map(self.cluster_to_persona_map)
        df["persona_name"] = df["persona_key"].apply(lambda k: PERSONAS_METADATA[k]["name"])
        df["action_playbook"] = df["persona_key"].apply(lambda k: PERSONAS_METADATA[k]["action_playbook"])

        self.fitted_ = True
        return df

    def _build_persona_mapping(self, clustered_df: pd.DataFrame) -> None:
        """Maps cluster IDs to business personas (VIP, Potential, At-Risk, Hibernating)."""
        summary = clustered_df.groupby("cluster_id").agg(
            mean_rec=("recency", "mean"),
            mean_freq=("frequency", "mean"),
            mean_mon=("monetary", "mean"),
        ).reset_index()

        # Ranking logic:
        # VIP: Highest frequency and monetary with low recency
        vip_cid = summary.sort_values(by=["mean_mon", "mean_freq"], ascending=False).iloc[0]["cluster_id"]

        # Hibernating: Highest recency and lowest frequency/monetary
        remaining = summary[summary["cluster_id"] != vip_cid]
        hib_cid = remaining.sort_values(by=["mean_rec"], ascending=False).iloc[0]["cluster_id"]

        # At-Risk vs Potential:
        # At-Risk has higher recency than Potential, but still significant monetary spend
        remaining2 = remaining[remaining["cluster_id"] != hib_cid]
        if len(remaining2) >= 2:
            at_risk_cid = remaining2.sort_values(by=["mean_rec", "mean_mon"], ascending=False).iloc[0]["cluster_id"]
            potential_cid = remaining2[remaining2["cluster_id"] != at_risk_cid].iloc[0]["cluster_id"]
        else:
            potential_cid = remaining2.iloc[0]["cluster_id"]
            at_risk_cid = potential_cid

        self.cluster_to_persona_map = {
            int(vip_cid): "vip",
            int(potential_cid): "potential",
            int(at_risk_cid): "at_risk",
            int(hib_cid): "hibernating",
        }

    def predict_single(
        self,
        recency: float,
        frequency: float,
        monetary: float,
    ) -> CustomerClusterPrediction:
        """Classifies an individual customer into a persona and playbook."""
        if not self.fitted_ or self.model is None:
            raise RuntimeError("KMeansSegmentation model must be fitted before prediction.")

        raw_vec = np.array([[float(recency), float(frequency), float(monetary)]])
        scaled_vec = self._preprocess_features(raw_vec, fit=False)

        cluster_id = int(self.model.predict(scaled_vec)[0])
        centroid = self.model.cluster_centers_[cluster_id]
        dist = float(np.linalg.norm(scaled_vec - centroid))

        persona_key = self.cluster_to_persona_map.get(cluster_id, "potential")
        meta = PERSONAS_METADATA[persona_key]

        return CustomerClusterPrediction(
            cluster_id=cluster_id,
            persona_key=persona_key,
            persona_name=meta["name"],
            action_playbook=meta["action_playbook"],
            base_churn_rate=meta["base_churn_rate"],
            distance_to_centroid=round(dist, 4),
        )

    def evaluate_k_range(
        self,
        rfm_df: pd.DataFrame,
        k_min: int = 2,
        k_max: int = 8,
    ) -> List[ClusteringEvaluation]:
        """Evaluates inertia, silhouette score, and Davies-Bouldin index across k in [k_min, k_max]."""
        raw_features = rfm_df[["recency", "frequency", "monetary"]].values.astype(float)
        X_scaled = self._preprocess_features(raw_features, fit=True)

        results: List[ClusteringEvaluation] = []
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(X_scaled)

            inertia = float(km.inertia_)
            sil = float(silhouette_score(X_scaled, labels))
            db = float(davies_bouldin_score(X_scaled, labels))

            results.append(ClusteringEvaluation(
                k=k,
                inertia=round(inertia, 2),
                silhouette=round(sil, 4),
                davies_bouldin=round(db, 4),
            ))

        return results
