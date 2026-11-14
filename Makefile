# PhoneSense Makefile. Windows users without make: see README "Plain commands".
PYTHON ?= python

.PHONY: setup test lint data data-quick sample handcheck handcheck-score eval all

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

eval:
	$(PYTHON) scripts/run_eval.py

all: test data eval
