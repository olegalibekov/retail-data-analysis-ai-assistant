# The web demo (src/agent/interface/web.py). The same image runs on Cloud Run or any Docker host.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src ./src
COPY config ./config
COPY data/golden_trios.json ./data/golden_trios.json

# Saved reports and traces are written here. They live as long as the container does.
RUN useradd --create-home app && mkdir -p logs && chown -R app /app
USER app

# Cloud Run provides PORT. One worker: conversations and sessions are kept in memory.
CMD ["sh", "-c", "exec uvicorn agent.interface.web:app --app-dir src --host 0.0.0.0 --port ${PORT:-8080}"]
