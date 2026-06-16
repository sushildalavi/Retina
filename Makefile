PYTHON ?= /opt/anaconda3/bin/python
NODE ?= node
NPM ?= npm
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

.PHONY: install test format lint prepare-data embeddings index eval recommendations random-baseline benchmark failures api frontend-install frontend demo all-local flickr8k-full-data flickr8k-full-embeddings flickr8k-full-index flickr8k-full-eval flickr8k-full-recommendations flickr8k-full-benchmark flickr8k-full-failures

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

recommendations:
	$(PYTHON) -m scripts.evaluate_recommendations --config $(CONFIG) $(REPORT_ARG)

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

frontend-install:
	cd frontend && $(NPM) install

frontend:
	@command -v $(NPM) >/dev/null 2>&1 || { echo "Node.js and npm are required. Install Node.js 20+ and run 'make frontend-install'."; exit 1; }
	@test -d frontend/node_modules || { echo "Frontend dependencies are missing. Run 'make frontend-install' first."; exit 1; }
	cd frontend && $(NPM) run dev

demo:
	$(PYTHON) -m scripts.run_gradio_demo --config $(CONFIG)

all-local:
	$(PYTHON) -m scripts.run_all_local --config $(CONFIG) $(if $(strip $(DATASET)) ,--dataset $(DATASET),) $(HF_DATASET_ARG) $(SAMPLE_SIZE_ARG) $(REPORT_ARG)

flickr8k-full-data:
	$(PYTHON) -m scripts.prepare_dataset --config $(CONFIG) --dataset hf_flickr8k --hf-dataset intro/flickr8k --sample-size full --report-prefix flickr8k_full_

flickr8k-full-embeddings:
	$(PYTHON) -m scripts.build_embeddings --config $(CONFIG) --report-prefix flickr8k_full_

flickr8k-full-index:
	$(PYTHON) -m scripts.build_image_text_index --config $(CONFIG)

flickr8k-full-eval:
	$(PYTHON) -m scripts.evaluate_retrieval --config $(CONFIG) --report-prefix flickr8k_full_

flickr8k-full-recommendations:
	$(PYTHON) -m scripts.evaluate_recommendations --config $(CONFIG) --report-prefix flickr8k_full_

flickr8k-full-benchmark:
	$(PYTHON) -m scripts.benchmark_embedding_runtime --config $(CONFIG) --report-prefix flickr8k_full_

flickr8k-full-failures:
	$(PYTHON) -m scripts.analyze_retrieval_failures --config $(CONFIG) --report-prefix flickr8k_full_
