.PHONY: install train-svm train-resnet train-vit test analyze clean

install:
	pip install -e .[dev]
	pre-commit install

train-svm:
	PYTHONPATH=. python src/main.py model=svm $(ARGS)

train-resnet:
	PYTHONPATH=. python src/main.py model=resnet18 $(ARGS)

train-vit:
	PYTHONPATH=. python src/main.py model=vit_b16 $(ARGS)

test:
	PYTHONPATH=. pytest tests/

analyze:
	PYTHONPATH=. python src/analyze_results.py

clean:
	rm -rf outputs/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +