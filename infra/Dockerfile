# Shared image for the api and ui services (compose profile "full").
FROM python:3.11-slim

ENV UV_SYSTEM_PYTHON=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --all-extras --no-dev

COPY . .
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
