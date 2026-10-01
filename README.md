# Clustering Algorithms Showcase

[![Python checks](https://github.com/PrashantSinghpns/clustering-algorithms-showcase/actions/workflows/ci.yml/badge.svg)](https://github.com/PrashantSinghpns/clustering-algorithms-showcase/actions/workflows/ci.yml)

Compare K-Means and DBSCAN on compact blobs and non-linear moons, with data-driven K selection, noise-aware diagnostics, and a DBSCAN sensitivity sweep.

## Run the experiment

Use Python 3.11 or 3.12. These experiments generate their own synthetic data and need no external download.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python src/run_clustering.py --output-dir outputs --eps 0.3 --min-samples 5
python -m pytest -q
```

For Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

## Method and interpretation

StandardScaler is fitted separately to 500 blobs (seed 0) and 500 moon samples (seed 42). Blob K is selected from 2–8 by maximum silhouette, without consulting the known synthetic labels. Two-cluster K-Means and DBSCAN are then compared on moons. Adjusted Rand Index uses synthetic ground truth only as a retrospective diagnostic; the sensitivity sweep does not auto-select parameters using ARI.

## Generated evidence

| Output under `outputs/` | Purpose |
|---|---|
| `k_selection_metrics.csv` | K-Means inertia and silhouette for valid candidate K values |
| `clustering_metrics.csv` | ARI, cluster count, noise fraction, and non-noise silhouette |
| `dbscan_sensitivity.csv` | Effects of changing eps while holding min_samples fixed |
| `metrics.json` | Selected K, dataset seeds, parameters, metrics, package versions |
| `kmeans_blobs.png`, `kmeans_moons.png`, `dbscan_moons.png` | Cluster assignments; DBSCAN noise is gray |

The command prints the measured report. Package versions are recorded because exact numerical values can vary across environments. Generated artifacts are ignored by git and can be recreated from the command above.

## Repository map

| Path | Purpose |
|---|---|
| `src/` | Runnable experiment and reusable functions |
| `tests/` | Shape, index, new-sample, or noise-handling checks |
| `notebooks/` | Original exploratory notebook; the script is the reproducible entry point |
| `.github/workflows/ci.yml` | Syntax and unit checks on Python 3.11/3.12 |

## Limitations

K-Means assumes compact groups; DBSCAN is sensitive to scaling, eps, and density differences. Silhouette excludes noise here and is null when fewer than two valid clusters remain. Compare it together with noise fraction; it is not a standalone score for the complete DBSCAN assignment.

Synthetic results demonstrate algorithm behavior; they do not establish performance on a business dataset. Apply held-out preprocessing when embedding these methods in a supervised workflow.


## Reference experiment

A locally executed synthetic reference run is recorded in [docs/reference_metrics.json](docs/reference_metrics.json), including the actual package versions. It is a reproducibility example, not a production benchmark.

| Dataset | Algorithm | ARI | Clusters | Noise fraction |
|---|---|---:|---:|---:|
| blobs | K-Means (k=3) | 1.0000 | 3 | 0.00% |
| moons | K-Means (k=2) | 0.4451 | 2 | 0.00% |
| moons | DBSCAN | 1.0000 | 2 | 0.00% |

On moons, K-Means had a higher silhouette despite poorer recovery of the known shape. This demonstrates why a single clustering score cannot replace inspection of geometry and domain context.
