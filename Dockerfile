FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY bot ./bot

RUN useradd --system --no-create-home botuser
USER botuser

CMD ["python", "main.py"]
