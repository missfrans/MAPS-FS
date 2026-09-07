from mapsfs.screening import select_d_optimal_blocks


def test_auto_doptimal_reaches_full_rank():
    cfg = {
        "feature_selection": {"enabled": ["a", "b", "c"]},
        "models": {"enabled": ["m1", "m2", "m3"]},
        "datasets": {"D": {"top_k_values": [5, 10, 20, 30]}},
        "screening": {"representation_blocks": "auto_min_rank"},
    }
    picked, summary = select_d_optimal_blocks(cfg, "D")
    assert len(picked) > 0
    assert summary["design_matrix_rank"] == summary["full_design_rank"]
