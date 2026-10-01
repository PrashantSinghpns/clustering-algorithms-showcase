# Clustering Algorithms Showcase

A reproducible comparison of K-Means and DBSCAN on synthetic datasets with compact and non-linear cluster shapes.

## What it demonstrates

- Scaling features before applying distance-based clustering.
- Selecting a K-Means cluster count with inertia and silhouette diagnostics.
- Using K-Means for compact, approximately spherical clusters.
- Showing why DBSCAN can better recover non-linear moon-shaped clusters.
- Evaluating cluster recovery with Adjusted Rand Index because synthetic ground truth is available for learning.

This is an algorithm demonstration, not a business prediction model. In real unsupervised problems, ground-truth labels are usually unavailable, so cluster quality needs domain review in addition to diagnostics.

## Correction from the original notebook

The original blob dataset contains three centres, while its final K-Means model used `n_clusters=2`. The cleaned implementation selects and fits `k=3` for that dataset. It keeps `k=2` for the two-moon comparison.

## Project structure

```text
src/run_clustering.py                    # Reproducible comparison pipeline
tests/test_clustering.py                 # K-selection checks
notebooks/clustering_exploration.ipynb   # Original exploratory notebook
```

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python src/run_clustering.py
pytest
```

The script writes diagnostic charts and CSV metrics to `outputs/`.
