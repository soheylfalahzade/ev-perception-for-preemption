.PHONY: install test lint typecheck eval sumo gif all
install:
	pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check src tests
typecheck:
	mypy src/evperc
eval:
	evperc-eval --dev 40 --test 100 --out results/eval
sumo:
	evperc-sumo --seeds 12 --jobs 2 --out results/sumo
gif:
	evperc-gif --out docs/assets/scenario.gif --seed 11 --results docs/results/results.json
all: lint typecheck test
