FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV CV_UI_PORT=7862
EXPOSE 7862

CMD ["python", "app.py"]
