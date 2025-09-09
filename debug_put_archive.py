#!/usr/bin/env python3
"""Debug put_archive issue."""

import docker
import tempfile
import os
import tarfile
import io

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
                print(f"Adding {file_path} as {arcname}")
                tar.add(file_path, arcname=arcname)
    
    tar_data.seek(0)
    print(f"Tar data size: {len(tar_data.getvalue())} bytes")
    
    # Create container
    container = client.containers.create(
        "python:3.9-slim",
        command="sleep 30",  # Keep container alive
        working_dir="/app",
        detach=True
    )
    
    # Start container
    container.start()
    
    # Try different paths
    print("Trying to put archive to /app:")
    put_result = container.put_archive("/app", tar_data.getvalue())
    print(f"Put archive result: {put_result}")
    
    # Debug: List files in container
    list_result = container.exec_run("ls -la /app")
    print(f"Files in /app: {list_result.output.decode('utf-8')}")
    
    # Try putting archive to root
    tar_data.seek(0)
    print("Trying to put archive to /:")
    put_result = container.put_archive("/", tar_data.getvalue())
    print(f"Put archive result: {put_result}")
    
    # Debug: List files in root
    list_result = container.exec_run("ls -la /")
    print(f"Files in /: {list_result.output.decode('utf-8')}")
    
    # Stop and remove container
    container.stop()
    container.remove()
    
except Exception as e:
    print("Error:", str(e))
    import traceback
    traceback.print_exc()
finally:
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)