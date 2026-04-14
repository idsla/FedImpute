#!/bin/bash
# Build the image
docker build --progress=plain -t fedimpute-app .

# Run basic_usage.py
echo "Running basic_usage.py..."
docker run --name temp-container fedimpute-app sh -c "python scripts/basic_usage.py | tee /app/logs/log1.txt"
docker cp temp-container:/app/logs/log1.txt ./logs/
docker rm temp-container

# Run real_scenario.py  
echo "Running real_scenario.py..."
docker run --name temp-container fedimpute-app sh -c "python scripts/real_scenario.py | tee /app/logs/log3.txt"
docker cp temp-container:/app/logs/log3.txt ./logs/
docker rm temp-container

# Run benchmark.py
# echo "Running benchmark.py..."
# docker run --name temp-container fedimpute-app sh -c "python scripts/benchmark.py | tee /app/logs/log2.txt"
# docker cp temp-container:/app/logs/log2.txt ./logs/
# docker rm temp-container