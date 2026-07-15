FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update
RUN apt-get install -y poppler-utils

COPY server.py .
COPY prompts.yaml .

EXPOSE 6060

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "6060", "--workers", "2", "--limit-concurrency", "10", "--backlog", "256", "--no-server-header" ]
