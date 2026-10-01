"""Compare K-Means and DBSCAN on datasets with different cluster shapes."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def scale(features: pd.DataFrame) -> pd.DataFrame:
    """Return standardized features while preserving readable column names."""
    scaled = StandardScaler().fit_transform(features)  # Scale each feature before distance-based clustering.
    return pd.DataFrame(scaled, columns=features.columns)


def choose_k(features: pd.DataFrame, maximum_k: int = 8) -> pd.DataFrame:
    """Calculate K-Means inertia and silhouette score for candidate cluster counts."""
    records = []
    for k in range(2, maximum_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(features)
        records.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(features, labels)})
    return pd.DataFrame(records)


def plot_clusters(data: pd.DataFrame, label_column: str, title: str, path: Path) -> None:
    """Save a scatter plot for a two-feature clustering result."""
    fig, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(data["feature_1"], data["feature_2"], c=data[label_column], cmap="tab10", s=22)
    axis.set(title=title, xlabel="feature_1", ylabel="feature_2")
    fig.colorbar(scatter, ax=axis, label="Cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    blob_features, blob_truth = make_blobs(
        n_samples=500, centers=3, n_features=2, cluster_std=0.60, random_state=0
    )
    blobs = pd.DataFrame(blob_features, columns=["feature_1", "feature_2"])
    blob_scaled = scale(blobs)
    selection = choose_k(blob_scaled)
    selection.to_csv(output_dir / "k_selection_metrics.csv", index=False)

    # The synthetic data has three compact groups, so K=3 follows the diagnostic curve.
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    blobs["kmeans_cluster"] = kmeans.fit_predict(blob_scaled)
    plot_clusters(blobs, "kmeans_cluster", "K-Means on compact blobs", output_dir / "kmeans_blobs.png")

    moon_features, moon_truth = make_moons(n_samples=500, noise=0.05, random_state=RANDOM_STATE)
    moons = pd.DataFrame(moon_features, columns=["feature_1", "feature_2"])
    moon_scaled = scale(moons)
    moons["kmeans_cluster"] = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(moon_scaled)
    moons["dbscan_cluster"] = DBSCAN(eps=0.3, min_samples=5).fit_predict(moon_scaled)
    plot_clusters(moons, "kmeans_cluster", "K-Means on non-linear moons", output_dir / "kmeans_moons.png")
    plot_clusters(moons, "dbscan_cluster", "DBSCAN on non-linear moons", output_dir / "dbscan_moons.png")

    metrics = pd.DataFrame(
        [
            {"dataset": "blobs", "algorithm": "K-Means (k=3)", "adjusted_rand_score": adjusted_rand_score(blob_truth, blobs["kmeans_cluster"])},
            {"dataset": "moons", "algorithm": "K-Means (k=2)", "adjusted_rand_score": adjusted_rand_score(moon_truth, moons["kmeans_cluster"])},
            {"dataset": "moons", "algorithm": "DBSCAN", "adjusted_rand_score": adjusted_rand_score(moon_truth, moons["dbscan_cluster"])},
        ]
    )
    metrics.to_csv(output_dir / "clustering_metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
