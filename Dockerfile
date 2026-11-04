FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Create a non-root user for security
RUN adduser --disabled-password --gecos "" --uid 1000 appuser
USER appuser

# Listens on 127.0.0.1:9000 by default, reachable only from containers in the same app
ENTRYPOINT ["python", "-m", "cosmos_agentic_retriever", "serve"]
