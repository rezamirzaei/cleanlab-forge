FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install -U pip && pip install -e ".[ui,notebooks,ai]"

EXPOSE 8501 8888

