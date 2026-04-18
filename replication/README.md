
# File Description

## Scripts Description

- `scripts/basic_usage.py`: A basic usage of the package.
- `scripts/benchmark.py`: A benchmark demonstration.
- `scripts/real_scenario.py`: A real scenario distributed imputation.

## Logs

- `logs/log1.txt`: output of `scripts/basic_usage.py`
- `logs/log2.txt`: output of `scripts/benchmark.py`
- `logs/log3.txt`: output of `scripts/real_scenario.py`

## Setup Scripts and Environment

- `dockerfile`: Docker image definition for reproducible runs.
- `run_docker.sh`: Build image and execute replication scripts in containers.
- `run_docker_prebuilt.sh`: Reuse an already-built local image and run scripts.
- `setup.sh`: Local (non-Docker) reproducible setup and optional run entrypoint.
- `requirements.txt`: Pinned Python dependency list used by both Docker and local setup.

# Instruction

## Run via Docker

Step 1: Install Docker: https://docs.docker.com/get-docker/

Step 2: Execute the scripts via Docker:

Approach 1: Build the image and run the scripts in one command:

```bash
bash run_docker.sh
```

Approach 2: Use pre-built image:

```bash
bash run_docker_prebuilt.sh
```

The logs will be saved in the `logs` folder.
The scripts force `linux/amd64` and single-thread numeric backends to reduce machine-to-machine drift.

## Run via Python

Step 1: Install Python (recommended: 3.12.3): https://www.python.org/downloads/release/python-3123/  
Note: other Python versions may run, but numerical results may not be exactly identical.

Step 2: Run reproducible setup (`run_target=none` by default):

```bash
bash setup.sh <run_target>
```

Available `run_target` values:
- `none`: setup only
- `basic`: run `scripts/basic_usage.py`
- `benchmark`: run `scripts/benchmark.py`
- `real`: run `scripts/real_scenario.py`
- `all`: run all three scripts

Examples:

```bash
bash setup.sh none .venv
bash setup.sh basic .venv
bash setup.sh all .venv
PYTHON_BIN=/path/to/python3.12.3 bash setup.sh all .venv
```

`setup.sh` installs `fedimpute==0.2.91`.

## (Optional): run the scripts manually with deterministic env vars.

Linux/macOS:
```bash
source .venv/bin/activate
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/basic_usage.py > logs/log1.txt
python scripts/benchmark.py > logs/log2.txt
python scripts/real_scenario.py > logs/log3.txt
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONHASHSEED="0"; $env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"; $env:VECLIB_MAXIMUM_THREADS="1"; $env:BLIS_NUM_THREADS="1"; $env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python scripts/basic_usage.py > logs/log1.txt
python scripts/benchmark.py > logs/log2.txt
python scripts/real_scenario.py > logs/log3.txt
```
