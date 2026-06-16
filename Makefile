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
	$(PYTHON) -m scripts.prepare_dataset --config configs/retina.yaml

embeddings:
	$(PYTHON) -m scripts.build_embeddings --config configs/retina.yaml

index:
	$(PYTHON) -m scripts.build_image_text_index --config configs/retina.yaml

eval:
	$(PYTHON) -m scripts.evaluate_retrieval --config configs/retina.yaml

benchmark:
	$(PYTHON) -m scripts.benchmark_embedding_runtime --config configs/retina.yaml

failures:
	$(PYTHON) -m scripts.analyze_retrieval_failures --config configs/retina.yaml

api:
	$(PYTHON) -m scripts.run_api --config configs/retina.yaml

demo:
	$(PYTHON) -m scripts.run_gradio_demo --config configs/retina.yaml

all-local:
	$(PYTHON) -m scripts.run_all_local --config configs/retina.yaml
