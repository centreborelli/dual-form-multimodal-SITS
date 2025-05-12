ruff:
	ruff check . --output-format pylint
mypy:
	mypy src/ tests/
test:
	pytest tests/
pylint:
	pylint --ignore=${PYLINT_IGNORED} src/ tests/ --load_plugins=perflint

check:
	ruff mypy test pylint
precommit:
	pre-commit install
createenv:
	pixi shell
initenv:
	pip install -e .
setup:
	createenv initenv precommit 
