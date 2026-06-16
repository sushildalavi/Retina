PYTHON ?= /opt/anaconda3/bin/python

.PHONY: install test format lint prepare-data embeddings index eval benchmark failures api demo all-local

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

format:
	$(PYTHON) -m black . || true

lint:
	$(PYTHON) -m ruff check . || true

prepare-data:
	$(PYTHON) scripts/prepare_dataset.py --config configs/retina.yaml

embeddings:
	$(PYTHON) scripts/build_embeddings.py --config configs/retina.yaml

index:
	$(PYTHON) scripts/build_image_text_index.py --config configs/retina.yaml

eval:
	$(PYTHON) scripts/evaluate_retrieval.py --config configs/retina.yaml

benchmark:
	$(PYTHON) scripts/benchmark_embedding_runtime.py --config configs/retina.yaml

failures:
	$(PYTHON) scripts/analyze_retrieval_failures.py --config configs/retina.yaml

api:
	$(PYTHON) scripts/run_api.py --config configs/retina.yaml

demo:
	$(PYTHON) scripts/run_gradio_demo.py --config configs/retina.yaml

all-local:
	$(PYTHON) scripts/run_all_local.py --config configs/retina.yaml

