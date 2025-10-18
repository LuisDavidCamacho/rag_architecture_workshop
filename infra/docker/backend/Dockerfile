FROM python:3.11-slim AS base

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION="1.8.4" \
    POETRY_VIRTUALENVS_IN_PROJECT="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    PYTHONUNBUFFERED="1"

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Install system build dependencies for Poetry and common Python packages.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl build-essential \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions first to leverage Docker layer caching.
COPY pyproject.toml poetry.lock* README.md ./

RUN poetry install --no-root --no-interaction

# Copy the application source.
COPY backend ./backend

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
