import pandas as pd
from mapsfs.refinement import choose_one_se_k


def test_one_se_selects_smallest_eligible_k():
    agg = pd.DataFrame({
        "top_k_num": [10, 20, 30, 40],
        "f1_mean": [0.960, 0.9695, 0.9710, 0.9708],
        "f1_se": [0.001, 0.001, 0.002, 0.001],
    })
    best_k, selected_k, best_mean, band = choose_one_se_k(agg)
    assert best_k == 30
    assert abs(band - 0.969) < 1e-12
    assert selected_k == 20
