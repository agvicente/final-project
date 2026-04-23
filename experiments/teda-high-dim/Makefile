.PHONY: install test experiments figures paper clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

experiments:
	python experiments/run_all.py

figures:
	python experiments/run_all.py --figures-only

paper:
	cd paper && latexmk -pdf main.tex

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	cd paper && latexmk -C
