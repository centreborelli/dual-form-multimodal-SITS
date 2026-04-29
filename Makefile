ruff_cmd:
	ruff check . --output-format pylint

mypy_cmd:
	mypy src/ tests/

test:
	pytest tests/

check: ruff_cmd mypy_cmd test

pylint_cmd:
	pylint --ignore=$(PYLINT_IGNORED) src/ tests/ --load-plugins=perflint

check: pylint_cmd ruff mypy test
