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

## Run via Python

Step 1: Install python 3.12.3 from official website: https://www.python.org/downloads/release/python-3123/

Step 2: Create a virtual environment and activate it:

Linux or mac
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows Powershell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Gitbash
```gitbash
python -m venv .venv
source .venv/Scripts/activate
```

Step 3: Install required packages:

```bash
pip install -r requirements.txt
pip install fedimpute==0.2.3
```

Step 4: Run the scripts:

```bash
python scripts/basic_usage.py > logs/log1.txt
python scripts/benchmark.py > logs/log2.txt
python scripts/real_scenario.py > logs/log3.txt
```

## Scripts Description

- `scripts/basic_usage.py`: A basic usage of the package.
- `scripts/benchmark.py`: A benchmark demonstration.
- `scripts/real_scenario.py`: A real scenario distributed imputation.

## Logs

- `logs/log{1,2,3}_docker.txt`: logs of running the scripts via Docker.
- `logs/log{1,2,3}_win.txt`: logs of running the scripts via Python on Windows.
- `logs/log{1,2,3}_linux.txt`: logs of running the scripts via Python on Linux.
