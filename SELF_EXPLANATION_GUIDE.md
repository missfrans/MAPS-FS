# MAPS-FS self-explanation output

MAPS-FS is designed to finish with an interpretable decision, not only a collection of CSV files.

After a one-command run, open:

```text
outputs/09_results_package/self_explanation/FINAL_SELF_EXPLANATION.md
```

Dataset-specific reports are written as both Markdown and JSON:

```text
<DATASET>_FINAL_RECOMMENDATION.md
<DATASET>_FINAL_RECOMMENDATION.json
```

## If the gate is NO

The report states that model-specific feature refinement was **not activated**, explains whether this occurred because the interaction block was not statistically supported or because the common representation remained statistically comparable across all downstream models, and reports the common feature-selection representation to retain.

A NO result is an intended framework outcome, not a failed experiment. MAPS-FS recommends retaining the common representation and choosing the downstream model from the Pareto-efficient operating points according to the application priority.

## If the gate is YES

The report states that model-specific refinement was activated, identifies the decision-relevant model(s), lists prioritized FS-model branches, reports branch-specific `best_k` and one-standard-error `selected_k`, and summarizes the final Pareto-efficient configurations.

## Objective-specific recommendations

MAPS-FS does not create an arbitrary weighted overall winner. Instead, the self-explanation identifies representatives for:

- highest F1;
- lowest false-alarm rate;
- lowest inference latency;
- smallest retained representation.

These are selected only from the Pareto-efficient final set. One configuration may represent several priorities simultaneously.

## Manuscript mode

With:

```bash
python run_mapsfs.py --config configs/mapsfs_template.yaml --mode manuscript
```

the final explanation additionally reports:

- practical MAPS-FS search evaluations;
- exhaustive-oracle evaluations;
- search saving;
- MAPS-minus-oracle best-F1 difference;
- oracle Pareto recall;
- MAPS Pareto precision;
- availability of repeated runtime profiling.

No arbitrary pass/fail threshold is imposed on the oracle metrics. The report presents the measured evidence so authors can interpret it in the context of the experimental protocol.

## Interpretation guardrails

The generated explanation deliberately avoids claims that are not supported by the framework:

- `not statistically distinguishable` is not called formal equivalence;
- Pareto membership is not called proof of real-world deployment superiority;
- objective-specific representatives are not called a universal best configuration;
- a gate NO result is not described as failure.

## Reference policy

Self-explanations compare final recommendations with `full` and the data-derived common decision `d0`. The final framework does not define or report an overlap subset.
