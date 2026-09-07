# MAPS-FS v1.3.0 — Audited OLS Gate release

This release synchronizes the executable repository with the Gate protocol used after the dependency-screening audit.

## Final Gate inference

Primary screening response: `f1`.

Reduced model:

```text
f1 ~ C(fs_method) + k_scaled + C(dl_model) + C(seed)
```

Full model:

```text
f1 ~ C(fs_method) + k_scaled + C(dl_model) + C(seed)
   + C(fs_method):C(dl_model)
   + k_scaled:C(dl_model)
```

The interaction block is evaluated with a nested-model partial F-test. The five seeds are treated as fixed experimental blocking effects.

The Gate returns `YES` only when:

```text
p_interaction < alpha
AND
common d0 is outside at least one model's Holm-adjusted comparable set
```

## Why this changed from v1.2

The earlier random-intercept MixedLM could report singular random-effect covariance and non-finite likelihood values when the estimated seed variance approached zero. The audited final protocol removes that numerical ambiguity by using the predefined seed set as a fixed blocking factor.

## Additional fixes

- comparable-set calculations now follow `statistics.screening_response` rather than being hard-coded to F1;
- positive and negative Gate regression controls were added;
- documentation and CLI wording were synchronized with the final OLS protocol;
- the no-overlap policy remains unchanged;
- D-optimal `auto_min_rank`, F1 primary Gate response, Holm correction, one-SE refinement, Pareto efficiency, and validation-only exhaustive oracle remain unchanged.

## Existing v1.2 experiment outputs

Changing the Gate inferential source code does not invalidate already completed feature-ranking or dependency-training runs. For an existing experiment, rerun only Stage 03 after updating the source, verify the Gate output, and then continue the downstream stages according to that audited decision. Do not mix Gate outputs produced by the obsolete non-finite MixedLM fit with the v1.3.0 result package.
