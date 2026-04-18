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

Step 1: Install python 3.12.3 from official website: https://www.python.org/downloads/release/python-3123/

Step 2: Run one-command reproducible setup:

```bash
bash setup.sh
```

Optional: use a custom python binary/path:

```bash
PYTHON_BIN=/path/to/python3.12.3 bash setup.sh .venv
```

This creates:
- `.venv`: virtual environment
- `.env.reproducible`: deterministic runtime env vars
- `run_local_repro.sh`: local reproducible runner

Step 3: Activate the virtual environment and deterministic env vars:

Linux or mac
```bash
source .venv/bin/activate
source .env.reproducible
```

Windows Powershell
```powershell
.venv\Scripts\Activate.ps1
# Use Git Bash to source .env.reproducible, or set these env vars manually in PowerShell.
```

Windows Gitbash
```gitbash
source .venv/Scripts/activate
source .env.reproducible
```

Step 4: Run scripts:

```bash
bash run_local_repro.sh .venv
```

## Scripts Description

- `scripts/basic_usage.py`: A basic usage of the package.
- `scripts/benchmark.py`: A benchmark demonstration.
- `scripts/real_scenario.py`: A real scenario distributed imputation.

## Logs

- `logs/log1.txt`: output of `scripts/basic_usage.py`
- `logs/log2.txt`: output of `scripts/benchmark.py`
- `logs/log3.txt`: output of `scripts/real_scenario.py`
