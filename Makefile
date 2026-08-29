.PHONY: train test eval-heldout inspect unit-test docker-build docker-run clean

BACKEND ?= lightning.gpu
DATA ?= data/nprint.csv
MODEL ?= model_out

train:
	python scripts/train.py $(DATA) $(MODEL) --backend $(BACKEND)

test:
	python scripts/test.py $(DATA) models/$(MODEL) --backend $(BACKEND)

eval-heldout:
	python scripts/eval_heldout.py $(DATA) models/$(MODEL) --backend $(BACKEND)

inspect:
	python scripts/inspect_model.py $(DATA) models/$(MODEL) --backend $(BACKEND)

unit-test:
	pytest tests/ -v

docker-build:
	docker compose build

docker-run:
	docker compose up quantum-train

clean:
	rm -rf models/*.pt models/*.json __pycache__ */__pycache__ */*/__pycache__
