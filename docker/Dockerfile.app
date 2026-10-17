# Streamlit UI image: only the HTTP client side, talks to the API container.
FROM python:3.13.14-slim-bookworm
RUN useradd --create-home --uid 10001 app \
    && pip install --no-cache-dir streamlit==1.66.0 httpx==0.28.1 pandas==3.0.6
WORKDIR /app
COPY --chown=app:app app ./app
USER app
ENV JOBNLP_API_URL=http://api:8000
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
