PYTHON ?= /opt/anaconda3/bin/python
CONFIG ?= configs/retina.yaml
REPORT_PREFIX ?=
DATASET ?= synthetic
HF_DATASET ?=
SAMPLE_SIZE ?= 0
MANIFEST ?=
OUTPUT_DIR ?=

REPORT_ARG = $(if $(strip $(REPORT_PREFIX)),--report-prefix $(REPORT_PREFIX),)
HF_DATASET_ARG = $(if $(strip $(HF_DATASET)),--hf-dataset $(HF_DATASET),)
SAMPLE_SIZE_ARG = $(if $(strip $(SAMPLE_SIZE)),--sample-size $(SAMPLE_SIZE),)
MANIFEST_ARG = $(if $(strip $(MANIFEST)),--manifest $(MANIFEST),)
OUTPUT_DIR_ARG = $(if $(strip $(OUTPUT_DIR)),--output-dir $(OUTPUT_DIR),)

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
	$(PYTHON) -m scripts.prepare_dataset --config $(CONFIG) --dataset $(DATASET) $(HF_DATASET_ARG) $(SAMPLE_SIZE_ARG) $(REPORT_ARG) $(MANIFEST_ARG) $(OUTPUT_DIR_ARG)

embeddings:
	$(PYTHON) -m scripts.build_embeddings --config $(CONFIG) $(REPORT_ARG)

index:
	$(PYTHON) -m scripts.build_image_text_index --config $(CONFIG)

eval:
	$(PYTHON) -m scripts.evaluate_retrieval --config $(CONFIG) $(REPORT_ARG)

random-baseline:
	$(PYTHON) -m scripts.evaluate_random_baseline --config $(CONFIG) $(REPORT_ARG)

benchmark:
	$(PYTHON) -m scripts.benchmark_embedding_runtime --config $(CONFIG) $(REPORT_ARG)

failures:
	$(PYTHON) -m scripts.analyze_retrieval_failures --config $(CONFIG) $(REPORT_ARG)

api:
	$(PYTHON) -m scripts.run_api --config $(CONFIG)

demo:
	$(PYTHON) -m scripts.run_gradio_demo --config $(CONFIG)

all-local:
	$(PYTHON) -m scripts.run_all_local --config $(CONFIG) $(if $(strip $(DATASET)) ,--dataset $(DATASET),) $(HF_DATASET_ARG) $(SAMPLE_SIZE_ARG) $(REPORT_ARG)
