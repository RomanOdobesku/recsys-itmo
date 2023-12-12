VENV := .venv

PROJECT := service
TESTS := tests
MODELS := rec_sys

IMAGE_NAME := reco_service
CONTAINER_NAME := reco_service

# Prepare

.venv:
	poetry lock
	poetry install --no-root
	poetry check

setup: .venv


# Clean

clean:
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf $(VENV)


# Format

isort_fix: .venv
	poetry run isort $(PROJECT) $(TESTS) $(MODELS)


black_fix:
	poetry run black $(PROJECT) $(TESTS) $(MODELS)

format: isort_fix black_fix


# Lint

isort: .venv
	poetry run isort --check $(PROJECT) $(TESTS) $(MODELS)

black:
	poetry run black --check --diff $(PROJECT) $(TESTS) $(MODELS)

flake: .venv
	poetry run flake8 $(PROJECT) $(TESTS) $(MODELS)

pylint: .venv
	poetry run pylint $(PROJECT) $(TESTS) $(MODELS)

lint: isort flake black pylint


# Test

.pytest:
	poetry run pytest $(TESTS)

test: .venv


# Docker

build:
	docker build . -t $(IMAGE_NAME)

run: build
	docker run -p 8080:8080 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Fix and check
fix: format lint test

# All

all: setup format lint test run

.DEFAULT_GOAL = all
