FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
COPY requirements.txt requirements-postgres.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-postgres.txt
COPY app ./app
COPY static ./static
COPY data/*.csv ./data/
EXPOSE 8000
CMD ["python", "-m", "app.main"]
