#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE_NAME="fedimpute-app"
PLATFORM="linux/amd64"
LOCAL_LOG_DIR="./logs"
DOCKERFILE_PATH="dockerfile"

mkdir -p "${LOCAL_LOG_DIR}"

cleanup() {
  docker rm -f temp-container >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Use an already-built local image. Fail fast if it does not exist.
docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 || {
  echo "Image '${IMAGE_NAME}' not found locally. Build it first with ./run_docker.sh or docker build -f ${DOCKERFILE_PATH} --platform=${PLATFORM} -t ${IMAGE_NAME} ." >&2
  exit 1
}

# Run basic_usage.py
cleanup
printf 'Running basic_usage.py...\n'
docker run --platform="${PLATFORM}" --name temp-container "${IMAGE_NAME}" \
  sh -c 'PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/basic_usage.py | tee /app/logs/log1.txt'
docker cp temp-container:/app/logs/log1.txt "${LOCAL_LOG_DIR}/"
docker rm temp-container >/dev/null

# Run real_scenario.py
cleanup
printf 'Running real_scenario.py...\n'
docker run --platform="${PLATFORM}" --name temp-container "${IMAGE_NAME}" \
  sh -c 'PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/real_scenario.py | tee /app/logs/log3.txt'
docker cp temp-container:/app/logs/log3.txt "${LOCAL_LOG_DIR}/"
docker rm temp-container >/dev/null

# Run benchmark.py
# cleanup
# printf 'Running benchmark.py...\n'
# docker run --platform="${PLATFORM}" --name temp-container "${IMAGE_NAME}" \
#   sh -c 'python scripts/benchmark.py | tee /app/logs/log2.txt'
# docker cp temp-container:/app/logs/log2.txt "${LOCAL_LOG_DIR}/"
# docker rm temp-container >/dev/null
