FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the index at image build time so the container starts ready to serve.
RUN python -m waterrag.indexer
EXPOSE 8000
CMD ["uvicorn", "waterrag.app:app", "--host", "0.0.0.0", "--port", "8000"]
