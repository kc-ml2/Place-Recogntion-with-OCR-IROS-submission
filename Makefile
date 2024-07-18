format:
	black .
	isort .

test:
	black . --check
	isort . --check-only
	env PYTHONPATH=. pytest --pylint -W ignore::DeprecationWarning
