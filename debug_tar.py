#!/usr/bin/env python3
"""Debug tar file creation."""

import tempfile
import os
import tarfile
import io

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

# List contents of tar file
tar_data.seek(0)
with tarfile.open(fileobj=tar_data, mode='r') as tar:
    print("Tar contents:")
    for member in tar.getmembers():
        print(f"  {member.name} ({member.size} bytes)")

# Clean up
import shutil
shutil.rmtree(temp_dir)