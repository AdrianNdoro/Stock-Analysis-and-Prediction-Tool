# Stock Analyzer dashboard. Build:  docker build -t stock-analyzer .
# Run (Windows Command Prompt, one line):
#   docker run --rm -p 8501:8501 -v stock-analyzer-data:/data stock-analyzer
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501 \
    CACHE_PATH=/data/market_cache.sqlite3 \
    LOG_JSON=true \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer

# Non-root user; /data holds the SQLite cache and is meant to be a volume.
RUN useradd --create-home --uid 10001 app \
    && mkdir /data \
    && chown app:app /data

WORKDIR /app

# Dependencies first, so code changes do not reinstall them (layer caching).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .streamlit ./.streamlit
COPY docker/healthcheck.py ./docker/healthcheck.py
COPY stock_analyzer ./stock_analyzer

USER app
VOLUME /data
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "docker/healthcheck.py"]

# Shell form of the port so platforms that inject $PORT (Render, Hugging Face) work unchanged.
CMD ["sh", "-c", "exec python -m streamlit run stock_analyzer/app/main.py --server.port=${PORT} --server.address=0.0.0.0"]
