FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements if you have one (optional but recommended)
# If you don't have requirements.txt yet, we'll write one below
COPY requirements.txt /app/

COPY /flask_app /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]