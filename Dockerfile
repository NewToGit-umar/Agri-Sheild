FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects PORT env var; gunicorn reads it at runtime
CMD gunicorn -b 0.0.0.0:$PORT app:app
