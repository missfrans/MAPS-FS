CONFIG ?= configs/mapsfs_template.yaml

.PHONY: test preflight rank screen gate refine select maps oracle validate-oracle results runtime reproduce-manuscript

test:
	pytest -q
	python -m compileall -q mapsfs scripts plugins tests

preflight:
	python scripts/00_preflight_validate.py --config $(CONFIG)

rank:
	python scripts/01_prepare_feature_rankings.py --config $(CONFIG)

screen:
	python scripts/02_run_dependency_screen.py --config $(CONFIG)

gate:
	python scripts/03_run_maps_gate.py --config $(CONFIG)

refine:
	python scripts/04_run_selective_refinement.py --config $(CONFIG)

select:
	python scripts/05_run_efficiency_selection.py --config $(CONFIG)

maps: preflight rank screen gate refine select results

oracle:
	python scripts/06_run_exhaustive_oracle.py --config $(CONFIG)

validate-oracle:
	python scripts/07_validate_against_oracle.py --config $(CONFIG)

results:
	python scripts/09_build_results_package.py --config $(CONFIG)

runtime:
	python scripts/08_runtime_profile.py --config $(CONFIG)

reproduce-manuscript:
	bash scripts/reproduce_manuscript_single_machine.sh
