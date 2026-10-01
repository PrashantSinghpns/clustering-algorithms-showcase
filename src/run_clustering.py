"""Compare cluster shape assumptions, model selection, and noise sensitivity."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def scale(features):
    return pd.DataFrame(StandardScaler().fit_transform(features), columns=features.columns, index=features.index)

def choose_k(features, maximum_k=8):
    """Score only valid silhouette candidates; identical points cannot define clusters."""
    upper = min(maximum_k, len(features) - 1, len(features.drop_duplicates()))
    if upper < 2:
        raise ValueError("K selection requires at least three samples and two distinct points.")
    records = []
    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels) if 1 < len(set(labels)) < len(labels) else np.nan
        records.append({"k": k, "inertia": model.inertia_, "silhouette": score})
    return pd.DataFrame(records)

def clustering_metrics(features, labels, truth):
    """Report noise explicitly; calculate silhouette only on non-noise clusters."""
    labels = np.asarray(labels)
    keep = labels != -1
    cluster_count = len(set(labels[keep]))
    silhouette = None
    if 1 < cluster_count < int(keep.sum()):
        silhouette = float(silhouette_score(features.loc[keep], labels[keep]))
    return {"clusters": cluster_count, "noise_fraction": float((~keep).mean()),
            "silhouette_non_noise": silhouette, "adjusted_rand_score": float(adjusted_rand_score(truth, labels))}

def plot_clusters(data, label_column, title, path):
    fig, axis = plt.subplots(figsize=(7, 5))
    for cluster, group in data.groupby(label_column):
        axis.scatter(group.feature_1, group.feature_2, s=22, alpha=0.8,
                     color="gray" if cluster == -1 else None,
                     label="Noise" if cluster == -1 else f"Cluster {cluster}")
    axis.set(title=title, xlabel="feature_1", ylabel="feature_2")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()
    if not np.isfinite(args.eps) or args.eps <= 0 or args.min_samples < 1:
        parser.error("eps must be finite and positive; min-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blob_features, blob_truth = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=0.6, random_state=0)
    blobs = pd.DataFrame(blob_features, columns=["feature_1", "feature_2"])
    scaled_blobs = scale(blobs)
    selection = choose_k(scaled_blobs)
    selection.to_csv(args.output_dir / "k_selection_metrics.csv", index=False)
    best_k = int(selection.loc[selection.silhouette.idxmax(), "k"])
    blobs["kmeans_cluster"] = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit_predict(scaled_blobs)
    plot_clusters(blobs, "kmeans_cluster", f"K-Means on blobs (selected k={best_k})", args.output_dir / "kmeans_blobs.png")
    moon_features, moon_truth = make_moons(n_samples=500, noise=0.05, random_state=RANDOM_STATE)
    moons = pd.DataFrame(moon_features, columns=["feature_1", "feature_2"])
    scaled_moons = scale(moons)
    moons["kmeans_cluster"] = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(scaled_moons)
    moons["dbscan_cluster"] = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(scaled_moons)
    records = []
    for dataset, algorithm, features, labels, truth in [
        ("blobs", f"K-Means (k={best_k})", scaled_blobs, blobs.kmeans_cluster, blob_truth),
        ("moons", "K-Means (k=2)", scaled_moons, moons.kmeans_cluster, moon_truth),
        ("moons", "DBSCAN", scaled_moons, moons.dbscan_cluster, moon_truth)]:
        records.append({"dataset": dataset, "algorithm": algorithm, **clustering_metrics(features, labels, truth)})
    for column, title, filename in [("kmeans_cluster", "K-Means on moons", "kmeans_moons.png"),
                                     ("dbscan_cluster", "DBSCAN on moons", "dbscan_moons.png")]:
        plot_clusters(moons, column, title, args.output_dir / filename)
    pd.DataFrame(records).to_csv(args.output_dir / "clustering_metrics.csv", index=False)
    sensitivity = []
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        labels = DBSCAN(eps=eps, min_samples=args.min_samples).fit_predict(scaled_moons)
        sensitivity.append({"eps": eps, **clustering_metrics(scaled_moons, labels, moon_truth)})
    pd.DataFrame(sensitivity).to_csv(args.output_dir / "dbscan_sensitivity.csv", index=False)
    report = {"selected_blob_k": best_k, "random_state": RANDOM_STATE, "blob_random_state": 0,
              "samples_per_dataset": 500, "eps": args.eps, "min_samples": args.min_samples,
              "metrics": records, "versions": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "matplotlib"]}}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
