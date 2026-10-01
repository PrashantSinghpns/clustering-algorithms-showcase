import numpy as np
import pandas as pd
import pytest
from src.run_clustering import choose_k, clustering_metrics, scale

def test_all_noise_has_no_silhouette():
    frame = pd.DataFrame({"x": [0., 1., 2.], "y": [2., 0., 1.]})
    metrics = clustering_metrics(frame, np.array([-1, -1, -1]), [0, 1, 2])
    assert metrics["clusters"] == 0
    assert metrics["noise_fraction"] == 1
    assert metrics["silhouette_non_noise"] is None

def test_small_dataset_caps_k_and_preserves_index():
    frame = pd.DataFrame({"x": [0., 1., 10., 11.], "y": [0., 1., 10., 11.]}, index=[8, 9, 10, 11])
    assert scale(frame).index.equals(frame.index)
    assert choose_k(frame).k.tolist() == [2, 3]
    with pytest.raises(ValueError):
        choose_k(frame.iloc[:2])
