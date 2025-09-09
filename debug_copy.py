#!/usr/bin/env python3
"""Simple test to debug Docker file copying."""

import docker
import tempfile
import os
import tarfile
import io
import time

# Create Docker client
client = docker.from_env()

try:
    # Create a temporary directory with a file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "main.py")
    
    with open(file_path, "w") as f:
        f.write("print('Hello, World!')")
    
    print(f"Created temp directory: {temp_dir}")
    print(f"Created file: {file_path}")
    print(f"File contents: {open(file_path).read()}")
    
    # Create a tar archive of the files
    tar_data = io.BytesIO()
    with tarfile.open(fileobj=tar_data, mode='w') as tar:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                tar.add(file_path, arcname=arcname)
    
    tar_data.seek(0)
    
    # Create container
    container = client.containers.create(
        "python:3.9-slim",
        command="sleep 30",  # Keep container alive
        working_dir="/app",
        detach=True
    )
    
    # Start container
    container.start()
    
    # Copy files to container
    container.put_archive("/app", tar_data.getvalue())
    
    # List files in container
    result = container.exec_run("ls -la /app")
    print("Files in container:")
    print(result.output.decode('utf-8'))
    
    # Execute the main code
    print("Executing code:")
    exec_result = container.exec_run("python /app/main.py")
    print("Exit code:", exec_result.exit_code)
    print("Output:", exec_result.output.decode('utf-8'))
    
    # Stop and remove container
    container.stop()
    container.remove()
    
except Exception as e:
    print("Error:", str(e))
finally:
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)