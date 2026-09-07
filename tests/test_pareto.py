import pandas as pd
from mapsfs.pareto import pareto_mask


def test_pareto_mask_four_objectives():
    df = pd.DataFrame({
        "f1": [0.90, 0.91, 0.89],
        "far": [0.02, 0.02, 0.03],
        "lat": [10.0, 9.0, 12.0],
        "k": [20, 20, 30],
    })
    mask = pareto_mask(df, {"f1":"max", "far":"min", "lat":"min", "k":"min"})
    assert mask.tolist() == [False, True, False]
