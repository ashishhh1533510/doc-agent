# Documentation Agent — Render deployment image
# Uses the Docker runtime (NOT Render's plain Python runtime) so we can install
# the Graphviz binary, which the diagrams library + png/svg/dot/drawio formats need.
FROM python:3.12-slim

# System deps:
#   graphviz -> required for png/svg/dot/drawio formats
#   git      -> GitPython shells out to the git binary to clone repo URLs
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Render injects $PORT at runtime — bind to it, never hardcode 8000.
CMD ["sh", "-c", "uvicorn doc_agent.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
