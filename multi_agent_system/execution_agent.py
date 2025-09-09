"""Execution agent for running and testing code in Docker containers."""

import asyncio
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import docker
from docker.errors import APIError, ContainerError, ImageNotFound

from .base_agent import BaseAgent
from .config import config
from .data_structures import AgentType, Response, Task, TaskType
from .logging_config import SystemLogger


class ExecutionAgent(BaseAgent):
    """Execution agent that runs code in secure Docker containers."""
    
    def __init__(self):
        super().__init__(AgentType.EXECUTION)
        self.docker_client = None
        self._setup_docker_client()
    
    def _setup_docker_client(self):
        """Setup Docker client."""
        try:
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize Docker client", error=str(e))
            raise
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        return {
            "Content-Type": "application/json"
        }
    
    def _get_model_name(self) -> str:
        """Get the model name used by this agent."""
        return "execution-agent"
    
    async def _make_api_call(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Execution agent doesn't make API calls, but processes tasks directly."""
        # This is a placeholder since the execution agent works directly with code
        # rather than calling an external API
        return {"content": prompt, "execution_result": "placeholder"}
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse API response."""
        return api_response.get("content", "")
    
    async def process_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Process an execution task and return results."""
        start_time = time.time()
        
        self.logger.log_task_start(
            task_id=task.id,
            agent_type=self.agent_type.value,
            task_type=task.type.value
        )
        
        try:
            # Extract code and execution instructions from task
            execution_plan = self._extract_execution_plan(task, context or {})
            
            # Execute the code in Docker container
            execution_result = await self._execute_in_docker(execution_plan)
            
            # Create response
            execution_time = time.time() - start_time
            response = Response(
                task_id=task.id,
                agent_type=self.agent_type,
                content=execution_result.get("output", ""),
                confidence_score=execution_result.get("confidence", 0.8),
                execution_time=execution_time,
                success=execution_result.get("success", False),
                metadata={
                    "execution_details": execution_result,
                    "docker_image": execution_plan.get("docker_image", "python:3.9-slim"),
                    "execution_environment": execution_plan.get("environment", {})
                },
                artifacts=execution_result.get("artifacts", [])
            )
            
            self.logger.log_task_completion(
                task_id=task.id,
                agent_type=self.agent_type.value,
                execution_time=execution_time,
                success=True
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            self.logger.error(
                "Task processing failed",
                task_id=task.id,
                agent_type=self.agent_type.value,
                error=error_message,
                execution_time=execution_time
            )
            
            return Response(
                task_id=task.id,
                agent_type=self.agent_type,
                content="",
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
    def _extract_execution_plan(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract execution plan from task and context."""
        self.logger.info(f"Extracting execution plan from task: {task} and context: {context}")
        
        execution_plan = {
            "docker_image": "python:3.9-slim",
            "code_files": {},
            "entry_point": "main.py",
            "requirements": [],
            "environment": {},
            "timeout": 300,  # 5 minutes default
            "tests": []
        }
        
        # Get context from task or from parameter
        task_context = {}
        if hasattr(task, 'context') and task.context:
            task_context = task.context
        elif context:
            task_context = context
            
        self.logger.info(f"Using context: {task_context}")
        
        # Parse task description for code and instructions
        task_content = f"{task.title}\n{task.description}\n{' '.join(task.requirements)}"
        
        # Extract code files from context or task content
        if task_context.get("code_files"):
            execution_plan["code_files"] = task_context["code_files"]
            self.logger.info(f"Found code_files in context: {task_context['code_files']}")
        elif task_context.get("code"):
            execution_plan["code_files"] = {"main.py": task_context["code"]}
            self.logger.info(f"Found code in context: {task_context['code']}")
        else:
            self.logger.info("No code found in context")
        
        # Extract requirements
        if task_context.get("requirements"):
            execution_plan["requirements"] = task_context["requirements"]
            self.logger.info(f"Found requirements in context: {task_context['requirements']}")
        elif "requirements.txt" in execution_plan["code_files"]:
            # Parse requirements from file
            req_content = execution_plan["code_files"]["requirements.txt"]
            execution_plan["requirements"] = [line.strip() for line in req_content.split("\n") if line.strip()]
            self.logger.info(f"Found requirements in requirements.txt: {execution_plan['requirements']}")
        
        # Extract entry point
        if task_context.get("entry_point"):
            execution_plan["entry_point"] = task_context["entry_point"]
            self.logger.info(f"Found entry_point in context: {task_context['entry_point']}")
        
        # Extract Docker image
        if task_context.get("docker_image"):
            execution_plan["docker_image"] = task_context["docker_image"]
            self.logger.info(f"Found docker_image in context: {task_context['docker_image']}")
        
        # Extract environment variables
        if task_context.get("environment"):
            execution_plan["environment"] = task_context["environment"]
            self.logger.info(f"Found environment in context: {task_context['environment']}")
        
        # Extract timeout
        if task_context.get("timeout"):
            execution_plan["timeout"] = task_context["timeout"]
            self.logger.info(f"Found timeout in context: {task_context['timeout']}")
        
        # Extract tests
        if task_context.get("tests"):
            execution_plan["tests"] = task_context["tests"]
            self.logger.info(f"Found tests in context: {task_context['tests']}")
        
        self.logger.info(f"Final execution plan: {execution_plan}")
        return execution_plan
    
    async def _execute_in_docker(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in a Docker container."""
        try:
            self.logger.info(f"Execution plan: {execution_plan}")
            
            # Build Docker image if it doesn't exist
            image_name = execution_plan["docker_image"]
            try:
                self.docker_client.images.get(image_name)
            except ImageNotFound:
                self.logger.info(f"Pulling Docker image: {image_name}")
                self.docker_client.images.pull(image_name)
            
            # Create a temporary directory with all files
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Write code files to temp directory
                self.logger.info(f"Creating files in temp directory: {temp_dir}")
                self.logger.info(f"Code files: {execution_plan['code_files']}")
                for filename, content in execution_plan["code_files"].items():
                    file_path = os.path.join(temp_dir, filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w") as f:
                        f.write(content)
                    self.logger.info(f"Created file: {file_path}")
                    self.logger.info(f"File content: {content}")
                
                # Log all files in temp directory
                self.logger.info("Files in temp directory:")
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        self.logger.info(f"  {file_path}")
                
                # Create requirements.txt if needed
                if execution_plan["requirements"]:
                    req_path = os.path.join(temp_dir, "requirements.txt")
                    with open(req_path, "w") as f:
                        f.write("\n".join(execution_plan["requirements"]))
                
                # Create a tar archive of the files
                import tarfile
                import io
                
                tar_data = io.BytesIO()
                with tarfile.open(fileobj=tar_data, mode='w') as tar:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            self.logger.info(f"Adding {file_path} as {arcname} to tar")
                            tar.add(file_path, arcname=arcname)
                
                tar_data.seek(0)
                self.logger.info(f"Tar data size: {len(tar_data.getvalue())} bytes")
                
                # Debug: List contents of tar file
                tar_data.seek(0)
                with tarfile.open(fileobj=tar_data, mode='r') as tar:
                    self.logger.info("Tar contents:")
                    for member in tar.getmembers():
                        self.logger.info(f"  {member.name} ({member.size} bytes)")
                
                tar_data.seek(0)
                
                # Create container
                container = self.docker_client.containers.create(
                    image_name,
                    command="sleep 300",  # Keep container alive
                    working_dir="/app",
                    detach=True
                )
                
                # Start container
                container.start()
                
                # Wait a moment for container to be ready
                time.sleep(1)
                
                # Copy files to container
                put_result = container.put_archive("/app", tar_data.getvalue())
                self.logger.info(f"Put archive result: {put_result}")
                
                # Debug: List files in container
                time.sleep(1)
                list_result = container.exec_run("ls -la /app")
                self.logger.info(f"Files in container: {list_result.output.decode('utf-8')}")
                
                # Install dependencies if requirements.txt exists
                if execution_plan["requirements"]:
                    install_cmd = "pip install -r requirements.txt"
                    self.logger.info("Installing dependencies in container")
                    dep_result = container.exec_run(
                        f"sh -c 'cd /app && {install_cmd}'"
                    )
                
                # Execute the main code
                self.logger.info(f"Executing code: {execution_plan['entry_point']}")
                exec_result = container.exec_run(f"python /app/{execution_plan['entry_point']}")
                self.logger.info(f"Execution exit code: {exec_result.exit_code}")
                logs = exec_result.output
                
                # Run tests if provided
                test_results = []
                if execution_plan["tests"]:
                    for test_file, test_content in execution_plan["tests"].items():
                        # Write test file to container
                        test_tar_data = io.BytesIO()
                        with tarfile.open(fileobj=test_tar_data, mode='w') as tar:
                            tarinfo = tarfile.TarInfo(name=test_file)
                            tarinfo.size = len(test_content)
                            tarinfo.mtime = time.time()
                            tar.addfile(tarinfo, io.BytesIO(test_content.encode('utf-8')))
                        
                        test_tar_data.seek(0)
                        container.put_archive("/app", test_tar_data.getvalue())
                        
                        # Run test
                        self.logger.info(f"Running test: {test_file}")
                        test_result = container.exec_run(f"python /app/{test_file}")
                        test_logs = test_result.output
                        
                        test_results.append({
                            "test_file": test_file,
                            "passed": test_result.exit_code == 0,
                            "output": test_logs.decode("utf-8") if isinstance(test_logs, bytes) else test_logs
                        })
                
                # Stop and remove container
                container.stop()
                container.remove()
                
                # Collect artifacts (this is a simplified approach)
                artifacts = []
                
                return {
                    "success": exec_result.exit_code == 0,
                    "output": logs.decode("utf-8") if isinstance(logs, bytes) else logs,
                    "test_results": test_results,
                    "artifacts": artifacts,
                    "confidence": 0.9 if exec_result.exit_code == 0 else 0.3
                }
            
            finally:
                # Clean up temporary directory
                import shutil
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            self.logger.error("Docker execution failed", error=str(e))
            return {
                "success": False,
                "output": str(e),
                "test_results": [],
                "artifacts": [],
                "confidence": 0.1
            }
    
    def _generate_execution_report(self, execution_result: Dict[str, Any]) -> str:
        """Generate a human-readable execution report."""
        report_lines = []
        
        if execution_result["success"]:
            report_lines.append("✅ Code execution completed successfully!")
        else:
            report_lines.append("❌ Code execution failed!")
        
        report_lines.append(f"Output:\n{execution_result['output']}")
        
        if execution_result["test_results"]:
            report_lines.append("\nTest Results:")
            for test in execution_result["test_results"]:
                status = "✅ PASSED" if test["passed"] else "❌ FAILED"
                report_lines.append(f"  {status} - {test['test_file']}")
                if not test["passed"]:
                    report_lines.append(f"    Output: {test['output']}")
        
        if execution_result["artifacts"]:
            report_lines.append(f"\nGenerated Artifacts ({len(execution_result['artifacts'])}):")
            for artifact in execution_result["artifacts"]:
                report_lines.append(f"  - {artifact}")
        
        return "\n".join(report_lines)
    
    async def health_check(self) -> bool:
        """Check if the execution agent is healthy."""
        try:
            # Check Docker connection
            if not self.docker_client:
                return False
            
            # Ping Docker daemon
            self.docker_client.ping()
            
            # Test with a simple container
            container = self.docker_client.containers.run(
                "hello-world",
                remove=True,
                detach=True,
                stdout=True,
                stderr=True
            )
            result = container.wait()
            return result["StatusCode"] == 0
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        await super().cleanup()
        # Docker client cleanup is handled automatically