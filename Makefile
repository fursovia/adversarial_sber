
.PHONY: check-codestyle codestyle

codestyle:
	pre-commit run --all-files

check-codestyle:
	python3 advsber/commands/verify.py --checks flake8
	python3 advsber/commands/verify.py --checks mypy
	python3 advsber/commands/verify.py --checks black
	python3 advsber/commands/verify.py --checks pytest
