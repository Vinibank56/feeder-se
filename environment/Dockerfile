FROM python:3.13-slim-bookworm

LABEL org.opencontainers.image.title="feeder-zip-se-agent-runtime"

RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.16.2
COPY data/ /app/data/
WORKDIR /app
