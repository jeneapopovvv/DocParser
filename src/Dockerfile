FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py prompts.yaml server.py document_store.py ./

EXPOSE 6060

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "6060", "--workers", "2", "--limit-concurrency", "10", "--backlog", "256", "--no-server-header"]
