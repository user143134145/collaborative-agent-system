#!/usr/bin/env python3
"""Simple test to debug Docker file creation."""

import docker

# Create Docker client
client = docker.from_env()

# Test script
script = """#!/bin/bash
set -e
cd /tmp
mkdir -p /tmp/app
cd /tmp/app
echo "print('Hello, World!')" > main.py
python main.py
"""

print("Running test script:")
print(script)

try:
    # Run the script in Docker
    logs = client.containers.run(
        "python:3.9-slim",
        command=f"sh -c \"{script}\"",
        remove=True
    )
    print("Success!")
    print("Output:", logs.decode("utf-8"))
except Exception as e:
    print("Error:", str(e))