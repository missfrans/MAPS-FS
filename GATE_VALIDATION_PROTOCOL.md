# MAPS-FS Gate validation protocol

The final MAPS-FS Gate uses nested OLS models with the five predefined seeds as fixed blocking effects. The reduced model is:

```text
response ~ C(fs_method) + k_scaled + C(dl_model) + C(seed)
```

The full model adds:

```text
C(fs_method):C(dl_model) + k_scaled:C(dl_model)
```

The interaction block is assessed with a nested-model partial F-test. The Gate returns `YES` only when `p_interaction < alpha` **and** the common representation `d0` is outside the Holm-adjusted statistically comparable set for at least one downstream model.

## Regression controls

`tests/test_gate.py` contains two deterministic synthetic controls:

- negative control: common response behavior -> expected Gate `NO`;
- positive control: strong model-specific representation behavior -> expected Gate `YES`.

A third test confirms that the within-model comparable-set analysis uses the configured response metric rather than a hard-coded F1 column.

These controls validate Gate logic. They are not substitutes for the real-data screening, oracle validation, or robustness analysis.
