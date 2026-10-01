import pandas as pd
from sklearn.datasets import make_blobs

from src.run_clustering import choose_k, scale


def test_k_selection_scores_each_candidate_count():
    features, _ = make_blobs(n_samples=60, centers=3, n_features=2, random_state=42)
    results = choose_k(scale(pd.DataFrame(features, columns=["feature_1", "feature_2"])), maximum_k=4)

    assert results["k"].tolist() == [2, 3, 4]
    assert results["silhouette"].between(-1, 1).all()
