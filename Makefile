PYTHON := .venv/bin/python

.PHONY: setup train demo

setup:
	python3.13 -m venv .venv
	.venv/bin/pip install -r requirements.txt

train:
	$(PYTHON) -m src.train

demo:
	$(PYTHON) -m src.demo
