import numpy as np
import pandas as pd

from mapsfs.statistics import (
    comparable_sets,
    interaction_block_test,
)


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

MODELS = [
    "cnn",
    "rnn",
    "dnn",
    "mlp",
    "lstm",
]

BLOCKS = [
    ("autoencoder", 5),
    ("lasso", 5),
    ("lasso", 40),
    ("mrmr", 20),
    ("rf", 25),
    ("rfe", 40),
]

REP_EFFECT = {
    ("autoencoder", 5):
        -0.020,

    ("lasso", 5):
        -0.015,

    ("lasso", 40):
        -0.010,

    ("mrmr", 20):
        -0.008,

    ("rf", 25):
        -0.005,

    ("rfe", 40):
        0.000,
}

MODEL_EFFECT = {
    "cnn":
        0.000,

    "rnn":
        -0.002,

    "dnn":
        0.001,

    "mlp":
        -0.001,

    "lstm":
        0.002,
}


def _make_control(
    positive: bool,
) -> pd.DataFrame:

    rng = (
        np.random.default_rng(
            1234
        )
    )

    rows = []

    for seed in SEEDS:

        seed_effect = (
            (seed - 44)
            * 0.0003
        )

        for model in MODELS:

            for (
                fs_method,
                top_k,
            ) in BLOCKS:

                interaction = 0.0

                if positive:

                    if (
                        model == "cnn"
                        and fs_method
                        == "autoencoder"
                        and top_k == 5
                    ):
                        interaction += (
                            0.055
                        )

                    if (
                        model == "lstm"
                        and fs_method
                        == "lasso"
                        and top_k == 40
                    ):
                        interaction += (
                            0.055
                        )

                    if (
                        model
                        in [
                            "rnn",
                            "dnn",
                            "mlp",
                        ]
                        and fs_method
                        == "rfe"
                        and top_k == 40
                    ):
                        interaction += (
                            0.015
                        )

                f1 = (
                    0.930
                    + REP_EFFECT[
                        (
                            fs_method,
                            top_k,
                        )
                    ]
                    + MODEL_EFFECT[
                        model
                    ]
                    + seed_effect
                    + interaction
                    + rng.normal(
                        0.0,
                        0.0012,
                    )
                )

                rows.append(
                    {
                        "seed":
                            seed,

                        "fs_method":
                            fs_method,

                        "top_k":
                            top_k,

                        "dl_model":
                            model,

                        "f1":
                            f1,
                    }
                )

    return pd.DataFrame(
        rows
    )


def _gate_from_metrics(
    df: pd.DataFrame,
):
    test, _, _ = (
        interaction_block_test(
            df,
            "f1",
        )
    )

    means = (
        df
        .groupby(
            [
                "fs_method",
                "top_k",
            ]
        )["f1"]
        .mean()
        .reset_index()
    )

    means["top_k_num"] = (
        pd.to_numeric(
            means["top_k"]
        )
    )

    d0row = (
        means
        .sort_values(
            [
                "f1",
                "top_k_num",
                "fs_method",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .iloc[0]
    )

    d0 = (
        str(
            d0row.fs_method
        ),
        str(
            d0row.top_k
        ),
    )

    _, comparable = (
        comparable_sets(
            df.assign(
                top_k=(
                    df["top_k"]
                    .astype(str)
                )
            ),
            alpha=0.05,
            method="paired_t",
            response="f1",
        )
    )

    decision_relevant = [
        model
        for model, candidates
        in comparable.items()
        if d0 not in candidates
    ]

    gate = bool(
        test[
            "p_interaction"
        ] < 0.05
        and decision_relevant
    )

    return (
        gate,
        test,
        d0,
        decision_relevant,
    )


def test_negative_control_returns_no():

    gate, test, d0, relevant = (
        _gate_from_metrics(
            _make_control(
                False
            )
        )
    )

    assert gate is False

    assert (
        test[
            "analysis_model"
        ]
        ==
        "OLS with seed fixed blocking effects"
    )

    assert np.isfinite(
        test[
            "f_statistic"
        ]
    )

    assert np.isfinite(
        test[
            "p_interaction"
        ]
    )

    assert (
        test[
            "p_interaction"
        ]
        >= 0.05
    )

    assert relevant == []

    assert d0 == (
        "rfe",
        "40",
    )


def test_positive_control_returns_yes():

    gate, test, d0, relevant = (
        _gate_from_metrics(
            _make_control(
                True
            )
        )
    )

    assert gate is True

    assert (
        test[
            "p_interaction"
        ]
        < 0.05
    )

    assert d0 == (
        "rfe",
        "40",
    )

    assert "cnn" in relevant
    assert "lstm" in relevant


def test_comparable_sets_follow_requested_response():

    df = _make_control(
        True
    )

    # Deliberately create another response whose
    # ordering differs from F1.
    df["roc_auc"] = (
        1.0
        - df["f1"]
    )

    comp, sets = (
        comparable_sets(
            df.assign(
                top_k=(
                    df["top_k"]
                    .astype(str)
                )
            ),
            alpha=0.05,
            method="paired_t",
            response="roc_auc",
        )
    )

    assert not comp.empty

    assert set(
        comp[
            "response_metric"
        ]
    ) == {
        "roc_auc"
    }

    assert set(
        sets
    ) == set(
        MODELS
    )
