"""Unit tests for K-Means Clustering and Persona Mapping."""

from src.models.kmeans import KMeansSegmentation


def test_kmeans_fitting_and_persona_mapping(sample_rfm_data):
    """Validates K-Means clustering fit, cluster assignment and persona mapping."""
    km = KMeansSegmentation(n_clusters=4, random_state=42)
    clustered = km.fit(sample_rfm_data)

    assert "cluster_id" in clustered.columns
    assert "persona_key" in clustered.columns
    assert "persona_name" in clustered.columns

    assigned_keys = set(clustered["persona_key"].unique())
    # Should identify multiple personas
    assert "vip" in assigned_keys or "potential" in assigned_keys


def test_kmeans_single_prediction(sample_rfm_data):
    """Tests classifying a single high-value customer profile."""
    km = KMeansSegmentation(n_clusters=4, random_state=42)
    km.fit(sample_rfm_data)

    # Typical VIP profile: recency 3 days, frequency 22, monetary 12000
    pred_vip = km.predict_single(recency=3, frequency=22, monetary=12000.0)
    assert pred_vip.cluster_id in [0, 1, 2, 3]
    assert pred_vip.persona_name != ""
    assert pred_vip.distance_to_centroid >= 0.0


def test_evaluate_k_range(sample_rfm_data):
    """Validates evaluation across k in [2, 5]."""
    km = KMeansSegmentation(n_clusters=4, random_state=42)
    evals = km.evaluate_k_range(sample_rfm_data, k_min=2, k_max=5)

    assert len(evals) == 4
    for ev in evals:
        assert ev.inertia > 0.0
        assert -1.0 <= ev.silhouette <= 1.0
        assert ev.davies_bouldin >= 0.0
