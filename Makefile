SHELL := /bin/sh

.PHONY: setup install precommit fmt lint test up down logs

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

install:
	pip install -r requirements.txt

precommit:
	pre-commit run --all-files

fmt:
	black . && isort .

lint:
	flake8 .

test:
	pytest -q

up:
	docker compose -f infra/docker-compose.yml up -d --build

down:
	docker compose -f infra/docker-compose.yml down -v

logs:
	docker compose -f infra/docker-compose.yml logs -f --tail=200
