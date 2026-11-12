# PhoneSense Makefile. Windows users without make: see README "Plain commands".
PYTHON ?= python

.PHONY: setup test lint data eval all

setup:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

data:
	$(PYTHON) scripts/build_data.py

eval:
	$(PYTHON) scripts/run_eval.py

all: test data eval
