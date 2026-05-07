FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by opencv-python
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the application using Cloud Run's dynamic PORT
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
