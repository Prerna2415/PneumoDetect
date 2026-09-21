FROM python:3.11-slim

# Libraries required by OpenCV
RUN apt-get update     && apt-get install -y --no-install-recommends libgl1 libglib2.0-0     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + trained weights (see .dockerignore)
COPY . .

# Render injects $PORT; 5000 is the local default
ENV PORT=5000
EXPOSE 5000

# gunicorn instead of Flask's development server. One worker + one thread
# keeps a single copy of the model in memory; the app serialises inference
# with a lock anyway. --timeout allows for slow CPU inference.
CMD gunicorn --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:${PORT} app.app:app
