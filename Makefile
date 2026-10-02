.PHONY: install run test lint up down logs smoke check-llm providers

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check .

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

smoke:
	python scripts/smoke_test.py --url http://localhost:8000

check-llm:
	python scripts/check_llm.py

providers:
	python scripts/check_llm.py --list
