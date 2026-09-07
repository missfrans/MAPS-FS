# PAIR-IDS migration note

This directory intentionally contains no active legacy analysis code. The MAPS-FS implementation replaces the PAIR-IDS post-hoc exhaustive ranking workflow with an evidence-triggered search workflow.

Legacy elements **not used as MAPS-FS core logic**:

- weighted composite pairing scores;
- normalization-dependent overall ranking;
- weight-sensitivity ranking as a selection mechanism;
- fixed F1/FAR tolerances for top-k selection;
- fixed top-20 selected-vs-full Wilcoxon analysis;
- automatic exhaustive FS x k x model evaluation as the practical method.

They may remain useful as historical or supplementary sensitivity analyses, but they should not be represented as MAPS-FS decision logic.
