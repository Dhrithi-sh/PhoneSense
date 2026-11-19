# PhoneSense Makefile. Windows users without make: see README "Plain commands".
PYTHON ?= python

.PHONY: setup test lint data data-quick sample handcheck handcheck-score index index-sample tfidf tune baselines tune-phrase t06-t07 bench eval all

setup:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

data:
	$(PYTHON) scripts/build_data.py

data-quick:
	$(PYTHON) scripts/build_data.py --limit 200000

sample:
	$(PYTHON) scripts/build_sample.py

handcheck:
	$(PYTHON) scripts/make_handcheck.py

handcheck-score:
	$(PYTHON) scripts/score_handcheck.py

index:
	$(PYTHON) scripts/build_index.py

index-sample:
	$(PYTHON) scripts/build_index.py --sample

tfidf: index
	$(PYTHON) scripts/run_tfidf.py --split all

tune: index
	$(PYTHON) scripts/tune_bm25.py

baselines: index
	$(PYTHON) scripts/run_baselines.py --split dev

tune-phrase: index
	$(PYTHON) scripts/tune_phrase.py

t06-t07: index
	$(PYTHON) scripts/run_t06_t07.py --split dev

bench: index
	$(PYTHON) scripts/bench_boolean.py

eval:
	$(PYTHON) scripts/run_eval.py

all: test data eval
