FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "multi_agentic_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
