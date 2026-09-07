from pathlib import Path

from mapsfs.oracle import oracle_design


def _cfg():
    return {
        "feature_selection": {"enabled": ["rfe", "rf"]},
        "models": {"enabled": ["dnn", "lstm"]},
        "datasets": {"D": {"top_k_values": [5, 10]}},
    }


def test_oracle_has_only_full_fixed_reference():
    design = oracle_design(_cfg(), "D")
    refs = design[design["reference"].astype(bool)]
    assert set(refs["fs_method"]) == {"full"}
    assert set(refs["top_k"].astype(str)) == {"all"}
    assert len(refs) == 2


def test_final_config_has_no_overlap_keys():
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "mapsfs_template.yaml").read_text(encoding="utf-8").lower()
    assert "overlap_features" not in text
    assert "include_overlap_reference" not in text
