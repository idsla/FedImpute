#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible local setup (non-Docker).
# Usage:
#   bash setup.sh
#   bash setup.sh <run_target>
#   bash setup.sh <run_target> <venv_dir>
#   PYTHON_BIN=/path/to/python bash setup.sh <run_target> <venv_dir>
# run_target:
#   none      setup only (default)
#   basic     run scripts/basic_usage.py
#   benchmark run scripts/benchmark.py
#   real      run scripts/real_scenario.py
#   all       run all three scripts

RUN_TARGET="${1:-none}"
VENV_DIR="${2:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi


# Create virtual environment.
echo "Creating virtual environment: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# Resolve venv paths for Unix and Windows.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  VENV_PYTHON="${VENV_DIR}/bin/python"
  VENV_PIP="${VENV_DIR}/bin/pip"
elif [[ -x "${VENV_DIR}/Scripts/python.exe" ]]; then
  VENV_PYTHON="${VENV_DIR}/Scripts/python.exe"
  VENV_PIP="${VENV_DIR}/Scripts/pip.exe"
else
  echo "Cannot locate venv python in ${VENV_DIR}" >&2
  exit 1
fi

# Pin installer toolchain + experiment dependencies.
echo "Installing pinned packaging tools..."
"${VENV_PIP}" install --no-cache-dir --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0
echo "Installing replication requirements..."
"${VENV_PIP}" install --no-cache-dir -r requirements.txt
"${VENV_PIP}" install --no-cache-dir fedimpute==0.2.91

# Deterministic runtime variables (hash + single-thread numeric libs).
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Optional run stage (uses the venv interpreter to avoid system-python drift).
mkdir -p ./logs
case "${RUN_TARGET}" in
  none)
    ;;
  basic)
    "${VENV_PYTHON}" scripts/basic_usage.py > logs/log1.txt
    ;;
  benchmark)
    "${VENV_PYTHON}" scripts/benchmark.py > logs/log2.txt
    ;;
  real)
    "${VENV_PYTHON}" scripts/real_scenario.py > logs/log3.txt
    ;;
  all)
    "${VENV_PYTHON}" scripts/basic_usage.py > logs/log1.txt
    "${VENV_PYTHON}" scripts/benchmark.py > logs/log2.txt
    "${VENV_PYTHON}" scripts/real_scenario.py > logs/log3.txt
    ;;
  *)
    echo "Invalid run_target: ${RUN_TARGET}" >&2
    echo "Use one of: none, basic, benchmark, real, all" >&2
    exit 1
    ;;
esac
