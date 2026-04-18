# Use an explicit Debian-based Python image and a fixed target platform for better cross-machine consistency.
FROM --platform=linux/amd64 python:3.12.3-bookworm

# Runtime settings to reduce sources of nondeterminism in Python and numeric libraries.
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /app

# Copy requirements first for better layer reuse.
COPY requirements.txt ./

# Upgrade packaging tools, then install dependencies.
RUN python -m pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fedimpute==0.2.91

# Copy all project files.
COPY . .

# Ensure log directory exists for tee output.
RUN mkdir -p /app/logs

# Default command can still be overridden by docker run.
# CMD ["python", "scripts/basic_usage.py"]
