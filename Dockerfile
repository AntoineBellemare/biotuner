# Use official Python image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for PortAudio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files to /app
COPY . /app

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install both basic and GUI dependencies from pyproject.toml
RUN pip install .[gui]

# Expose Cloud Run's required port (8080)
EXPOSE 8080

# Set environment variable for Streamlit
ENV PORT 8080

# Run Streamlit app when the container starts
CMD ["streamlit", "run", "app/gui.py", "--server.port=8080", "--server.address=0.0.0.0"]
