FROM python:3.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install pytest==8.4.1 pytest-json-ctrf==0.3.5 numpy==2.2.6
COPY . /tests/
RUN sed -i 's/\r$//' /tests/test.sh && chmod +x /tests/test.sh
RUN mkdir -p /app /logs/verifier
