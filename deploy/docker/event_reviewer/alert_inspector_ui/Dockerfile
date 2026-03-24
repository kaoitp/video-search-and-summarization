FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Port 7860 — Gradio UI  |  8500 — kept for compose compatibility
EXPOSE 7860 8500

CMD ["python", "app.py"]
