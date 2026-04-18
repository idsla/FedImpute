#!/usr/bin/env bash
set -Eeuo pipefail

# Deterministic local (non-Docker) setup for replication.
# Usage:
#   bash setup.sh
#   bash setup.sh .venv

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRED_PYTHON_VERSION="3.12.3"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if [[ "${PYTHON_VERSION}" != "${REQUIRED_PYTHON_VERSION}" ]]; then
  echo "Expected Python ${REQUIRED_PYTHON_VERSION}, got ${PYTHON_VERSION}." >&2
  echo "Set PYTHON_BIN to a Python 3.12.3 executable, for example:" >&2
  echo "  PYTHON_BIN=/path/to/python3.12.3 bash setup.sh" >&2
  exit 1
fi

echo "Creating virtual environment: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

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

echo "Installing pinned packaging tools..."
"${VENV_PIP}" install --no-cache-dir --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0

echo "Installing replication requirements..."
"${VENV_PIP}" install --no-cache-dir -r requirements.txt
"${VENV_PIP}" install --no-cache-dir fedimpute==0.2.7

cat > .env.reproducible <<'EOF'
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
EOF

cat > run_local_repro.sh <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

VENV_DIR="${1:-.venv}"

if [[ -x "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
elif [[ -x "${VENV_DIR}/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/Scripts/activate"
else
  echo "Cannot find activation script in ${VENV_DIR}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source ./.env.reproducible

mkdir -p ./logs

python scripts/basic_usage.py > logs/log1.txt
python scripts/benchmark.py > logs/log2.txt
python scripts/real_scenario.py > logs/log3.txt
EOF
chmod +x run_local_repro.sh

echo ""
echo "Setup completed."
echo "Next:"
echo "  source ${VENV_DIR}/bin/activate        # Linux/macOS"
echo "  or ${VENV_DIR}\\Scripts\\Activate.ps1   # Windows PowerShell"
echo "  source ./.env.reproducible"
echo "  bash run_local_repro.sh ${VENV_DIR}"
echo ""
echo "Installed package versions:"
"${VENV_PYTHON}" -m pip freeze | grep -E '^(fedimpute|numpy|pandas|scipy|scikit-learn|torch|statsmodels|xgboost)=='
