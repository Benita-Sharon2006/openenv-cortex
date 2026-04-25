FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openenv-core fastapi uvicorn pydantic

COPY . .

RUN pip install --no-cache-dir -e .

ENV ENABLE_WEB_INTERFACE=true
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]