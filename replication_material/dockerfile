# Use exact Python version for reproducibility
FROM python:3.12.3

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fedimpute==0.2.3

# Copy all project files
COPY . .

# # Default command (you can override this when running)
# CMD ["python", "basic_usage.py"]