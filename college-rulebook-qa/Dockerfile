FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8501
# Pass your key at run time:  docker run -p 8501:8501 -e GOOGLE_API_KEY=... rulebook-qa
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
