FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md config.py ./
COPY rag_core ./rag_core
COPY rag_server ./rag_server
COPY rag_cli ./rag_cli

RUN pip install --upgrade pip && pip install .

EXPOSE 8501

CMD ["python", "-m", "rag_cli.main", "server", "--host", "0.0.0.0", "--port", "8501"]
