#!/usr/bin/env python3
"""Simple test to debug Docker volume mounting."""

import docker
import tempfile
import os

# Create Docker client
client = docker.from_env()

# Create temporary directory and file
temp_dir = tempfile.mkdtemp()
file_path = os.path.join(temp_dir, "main.py")

with open(file_path, "w") as f:
    f.write("print('Hello, World!')")

print(f"Created temp directory: {temp_dir}")
print(f"Created file: {file_path}")
print(f"File contents: {open(file_path).read()}")

try:
    # Run the script in Docker
    logs = client.containers.run(
        "python:3.9-slim",
        command="python /app/main.py",
        volumes={temp_dir: {"bind": "/app", "mode": "rw"}},
        working_dir="/app",
        remove=True
    )
    print("Success!")
    print("Output:", logs.decode("utf-8"))
except Exception as e:
    print("Error:", str(e))
finally:
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)