.PHONY: run test rebuild

run:
	docker compose up --build

test:
	python -m pytest tests -q

rebuild:
	docker compose exec app python -m rag_cli.main index --rebuild
