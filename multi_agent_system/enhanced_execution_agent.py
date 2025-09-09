"""Enhanced Execution agent for fully autonomous coding with Docker integration."""

import asyncio
import json
import os
import tempfile
import time
import shutil
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import docker
from docker.errors import APIError, ContainerError, ImageNotFound

from .base_agent import BaseAgent
from .config import config
from .data_structures import AgentType, Response, Task, TaskType
from .logging_config import SystemLogger


class EnhancedExecutionAgent(BaseAgent):
    """Enhanced execution agent that runs code in secure Docker containers with full autonomy."""

    def __init__(self):
        super().__init__(AgentType.EXECUTION)
        self.docker_client = None
        self.supported_images = {
            "python": "python:3.9-slim",
            "node": "node:18-alpine",
            "go": "golang:1.19-alpine",
        }
        self._setup_docker_client()
        self.orchestrator = None  # Reference to the orchestrator for checkpoint access

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
        return "enhanced-execution-agent"

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
                    "execution_environment": execution_plan.get("environment", {}),
                    "project_structure": execution_plan.get("project_structure", {}),
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
        self.logger.info(f"Extracting execution plan from task: {task.title} and context")

        execution_plan = {
            "docker_image": "python:3.9-slim",
            "code_files": {},
            "entry_point": "main.py",
            "requirements": [],
            "environment": {},
            "timeout": 300,  # 5 minutes default
            "tests": {},
            "project_structure": {},
            "build_commands": [],
            "runtime_commands": [],
            "language": "python"
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
            self.logger.info(f"Found code_files in context: {len(task_context['code_files'])} files")
        elif task_context.get("code"):
            execution_plan["code_files"] = {"main.py": task_context["code"]}
            self.logger.info("Found code in context")

        # Extract project structure
        if task_context.get("project_structure"):
            execution_plan["project_structure"] = task_context["project_structure"]
            self.logger.info("Found project structure in context")

        # Extract requirements
        if task_context.get("requirements"):
            execution_plan["requirements"] = task_context["requirements"]
            self.logger.info(f"Found requirements in context: {len(task_context['requirements'])} items")
        elif "requirements.txt" in execution_plan["code_files"]:
            # Parse requirements from file
            req_content = execution_plan["code_files"]["requirements.txt"]
            execution_plan["requirements"] = [line.strip() for line in req_content.split("\n") if line.strip()]
            self.logger.info(f"Found requirements in requirements.txt: {len(execution_plan['requirements'])} items")

        # Extract entry point
        if task_context.get("entry_point"):
            execution_plan["entry_point"] = task_context["entry_point"]
            self.logger.info(f"Found entry_point in context: {task_context['entry_point']}")

        # Extract Docker image
        if task_context.get("docker_image"):
            execution_plan["docker_image"] = task_context["docker_image"]
            self.logger.info(f"Found docker_image in context: {task_context['docker_image']}")
        elif task_context.get("language"):
            language = task_context["language"].lower()
            if language in self.supported_images:
                execution_plan["docker_image"] = self.supported_images[language]
                execution_plan["language"] = language
                self.logger.info(f"Using language-specific image: {execution_plan['docker_image']}")

        # Extract environment variables
        if task_context.get("environment"):
            execution_plan["environment"] = task_context["environment"]
            self.logger.info(f"Found environment in context: {len(task_context['environment'])} variables")

        # Extract timeout
        if task_context.get("timeout"):
            execution_plan["timeout"] = task_context["timeout"]
            self.logger.info(f"Found timeout in context: {task_context['timeout']}")

        # Extract tests
        if task_context.get("tests"):
            execution_plan["tests"] = task_context["tests"]
            self.logger.info(f"Found tests in context: {len(task_context['tests'])} test files")

        # Extract build commands
        if task_context.get("build_commands"):
            execution_plan["build_commands"] = task_context["build_commands"]
            self.logger.info(f"Found build commands: {len(task_context['build_commands'])} commands")

        # Extract runtime commands
        if task_context.get("runtime_commands"):
            execution_plan["runtime_commands"] = task_context["runtime_commands"]
            self.logger.info(f"Found runtime commands: {len(task_context['runtime_commands'])} commands")

        self.logger.info(f"Final execution plan: {execution_plan}")
        return execution_plan

    async def _execute_in_docker(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in a Docker container with full autonomy."""
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
                self.logger.info(f"Code files: {list(execution_plan['code_files'].keys())}")
                
                for filename, content in execution_plan["code_files"].items():
                    file_path = os.path.join(temp_dir, filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w") as f:
                        f.write(content)
                    self.logger.info(f"Created file: {file_path}")

                # Create project structure if specified
                if execution_plan["project_structure"]:
                    self._create_project_structure(temp_dir, execution_plan["project_structure"])

                # Create requirements.txt if needed
                if execution_plan["requirements"]:
                    req_path = os.path.join(temp_dir, "requirements.txt")
                    with open(req_path, "w") as f:
                        f.write("\n".join(execution_plan["requirements"]))
                    self.logger.info(f"Created requirements.txt with {len(execution_plan['requirements'])} dependencies")

                # Create a tar archive of the files
                import tarfile
                import io

                tar_data = io.BytesIO()
                with tarfile.open(fileobj=tar_data, mode='w') as tar:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            tar.add(file_path, arcname=arcname)

                tar_data.seek(0)
                self.logger.info(f"Tar data size: {len(tar_data.getvalue())} bytes")

                # Create container
                container = self.docker_client.containers.create(
                    image_name,
                    command="sleep 600",  # Keep container alive for 10 minutes
                    working_dir="/app",
                    detach=True,
                    environment=execution_plan.get("environment", {})
                )

                # Start container
                container.start()

                # Wait a moment for container to be ready
                time.sleep(2)

                # Copy files to container
                put_result = container.put_archive("/app", tar_data.getvalue())
                self.logger.info(f"Put archive result: {put_result}")

                # Debug: List files in container
                time.sleep(1)
                list_result = container.exec_run("ls -laR /app")
                self.logger.info(f"Files in container:\n{list_result.output.decode('utf-8')}")

                # Install dependencies if requirements.txt exists
                install_result = None
                if execution_plan["requirements"]:
                    install_cmd = self._get_install_command(execution_plan["language"])
                    self.logger.info(f"Installing dependencies with: {install_cmd}")
                    install_result = container.exec_run(
                        f"sh -c 'cd /app && {install_cmd}'",
                        stdout=True,
                        stderr=True
                    )
                    self.logger.info(f"Install result exit code: {install_result.exit_code}")
                    if install_result.exit_code != 0:
                        self.logger.error(f"Install failed: {install_result.output.decode('utf-8')}")

                # Run build commands if specified
                build_results = []
                if execution_plan["build_commands"]:
                    for cmd in execution_plan["build_commands"]:
                        self.logger.info(f"Running build command: {cmd}")
                        build_result = container.exec_run(
                            f"sh -c 'cd /app && {cmd}'",
                            stdout=True,
                            stderr=True
                        )
                        build_results.append({
                            "command": cmd,
                            "exit_code": build_result.exit_code,
                            "output": build_result.output.decode("utf-8") if isinstance(build_result.output, bytes) else build_result.output
                        })
                        self.logger.info(f"Build command '{cmd}' exit code: {build_result.exit_code}")

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
                        test_cmd = self._get_test_command(execution_plan["language"], test_file)
                        test_result = container.exec_run(test_cmd, stdout=True, stderr=True)
                        test_logs = test_result.output

                        test_results.append({
                            "test_file": test_file,
                            "passed": test_result.exit_code == 0,
                            "exit_code": test_result.exit_code,
                            "output": test_logs.decode("utf-8") if isinstance(test_logs, bytes) else test_logs
                        })
                        self.logger.info(f"Test '{test_file}' passed: {test_result.exit_code == 0}")

                # Run runtime commands if specified
                runtime_results = []
                if execution_plan["runtime_commands"]:
                    for cmd in execution_plan["runtime_commands"]:
                        self.logger.info(f"Running runtime command: {cmd}")
                        runtime_result = container.exec_run(
                            f"sh -c 'cd /app && {cmd}'",
                            stdout=True,
                            stderr=True
                        )
                        runtime_results.append({
                            "command": cmd,
                            "exit_code": runtime_result.exit_code,
                            "output": runtime_result.output.decode("utf-8") if isinstance(runtime_result.output, bytes) else runtime_result.output
                        })
                        self.logger.info(f"Runtime command '{cmd}' exit code: {runtime_result.exit_code}")

                # Collect artifacts (list all files in the container)
                artifacts = []
                try:
                    artifact_result = container.exec_run("find /app -type f")
                    if artifact_result.exit_code == 0:
                        artifacts = artifact_result.output.decode("utf-8").strip().split("\n")
                        artifacts = [a for a in artifacts if a.startswith("/app/")]
                except Exception as e:
                    self.logger.warning(f"Failed to collect artifacts: {e}")

                # Stop and remove container
                container.stop()
                container.remove()

                # Generate execution report
                execution_report = self._generate_execution_report({
                    "success": exec_result.exit_code == 0,
                    "output": logs.decode("utf-8") if isinstance(logs, bytes) else logs,
                    "install_result": install_result,
                    "build_results": build_results,
                    "test_results": test_results,
                    "runtime_results": runtime_results,
                    "artifacts": artifacts
                })

                return {
                    "success": exec_result.exit_code == 0,
                    "output": execution_report,
                    "raw_output": logs.decode("utf-8") if isinstance(logs, bytes) else logs,
                    "install_result": install_result,
                    "build_results": build_results,
                    "test_results": test_results,
                    "runtime_results": runtime_results,
                    "artifacts": artifacts,
                    "confidence": self._calculate_confidence(exec_result.exit_code, test_results)
                }

            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            self.logger.error("Docker execution failed", error=str(e))
            return {
                "success": False,
                "output": str(e),
                "install_result": None,
                "build_results": [],
                "test_results": [],
                "runtime_results": [],
                "artifacts": [],
                "confidence": 0.1
            }

    def _create_project_structure(self, base_dir: str, structure: Dict[str, Any]) -> None:
        """Create project directory structure."""
        for path, content in structure.items():
            full_path = os.path.join(base_dir, path)
            if isinstance(content, dict):
                # Create directory
                os.makedirs(full_path, exist_ok=True)
                # Recursively create sub-structure
                self._create_project_structure(full_path, content)
            else:
                # Create file with content
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content if isinstance(content, str) else str(content))
                self.logger.info(f"Created file with content: {full_path}")

    def _get_install_command(self, language: str) -> str:
        """Get the appropriate install command for the language."""
        install_commands = {
            "python": "pip install --no-cache-dir -r requirements.txt",
            "node": "npm install",
            "go": "go mod tidy"
        }
        return install_commands.get(language, "echo 'No install command for language'")

    def _get_execution_command(self, language: str, entry_point: str) -> str:
        """Get the appropriate execution command for the language."""
        execution_commands = {
            "python": f"python {entry_point}",
            "node": f"node {entry_point}",
            "go": f"go run {entry_point}"
        }
        return execution_commands.get(language, f"python {entry_point}")

    def _get_test_command(self, language: str, test_file: str) -> str:
        """Get the appropriate test command for the language."""
        test_commands = {
            "python": f"python -m pytest {test_file}" if "test_" in test_file else f"python {test_file}",
            "node": f"npm test" if test_file == "package.json" else f"node {test_file}",
            "go": f"go test -v ./{test_file}" if test_file.endswith("_test.go") else f"go test -v ./..."
        }
        return test_commands.get(language, f"python {test_file}")

    def _calculate_confidence(self, exit_code: int, test_results: List[Dict]) -> float:
        """Calculate confidence score based on execution results."""
        # Base confidence from execution success
        base_confidence = 0.9 if exit_code == 0 else 0.3
        
        # Adjust based on test results
        if test_results:
            passed_tests = sum(1 for t in test_results if t["passed"])
            total_tests = len(test_results)
            test_success_rate = passed_tests / total_tests if total_tests > 0 else 0
            # Weight test results as 40% of confidence
            base_confidence = base_confidence * 0.6 + test_success_rate * 0.4
            
        return min(base_confidence, 1.0)

    def _generate_execution_report(self, execution_result: Dict[str, Any]) -> str:
        """Generate a comprehensive execution report."""
        report_lines = []
        
        # Overall status
        if execution_result["success"]:
            report_lines.append("✅ Code execution completed successfully!")
        else:
            report_lines.append("❌ Code execution failed!")
        
        # Raw output
        raw_output = execution_result.get("raw_output", "")
        if raw_output:
            report_lines.append(f"\n📝 Execution Output:")
            report_lines.append("-" * 30)
            # Limit output to 1000 chars to prevent overly long reports
            output_preview = raw_output[:1000] + ("..." if len(raw_output) > 1000 else "")
            report_lines.append(output_preview)
            report_lines.append("-" * 30)
        
        # Installation results
        install_result = execution_result.get("install_result")
        if install_result:
            report_lines.append(f"\n📦 Dependency Installation:")
            report_lines.append(f"   Exit Code: {install_result.exit_code}")
            if install_result.exit_code != 0:
                report_lines.append("   ❌ Failed")
                install_output = install_result.output.decode("utf-8") if isinstance(install_result.output, bytes) else install_result.output
                if install_output:
                    report_lines.append(f"   Error: {install_output[:200]}...")
            else:
                report_lines.append("   ✅ Success")
        
        # Build results
        build_results = execution_result.get("build_results", [])
        if build_results:
            report_lines.append(f"\n🏗️  Build Results ({len(build_results)} commands):")
            for build in build_results:
                status = "✅" if build["exit_code"] == 0 else "❌"
                report_lines.append(f"   {status} {build['command']}")
                if build["exit_code"] != 0:
                    report_lines.append(f"      Error: {build['output'][:200]}...")
        
        # Test results
        test_results = execution_result.get("test_results", [])
        if test_results:
            passed = sum(1 for t in test_results if t["passed"])
            total = len(test_results)
            report_lines.append(f"\n🧪 Test Results ({passed}/{total} passed):")
            for test in test_results:
                status = "✅ PASSED" if test["passed"] else "❌ FAILED"
                report_lines.append(f"   {status} - {test['test_file']}")
                if not test["passed"]:
                    report_lines.append(f"      Exit Code: {test['exit_code']}")
                    if test["output"]:
                        report_lines.append(f"      Output: {test['output'][:200]}...")
        
        # Runtime results
        runtime_results = execution_result.get("runtime_results", [])
        if runtime_results:
            report_lines.append(f"\n🚀 Runtime Results ({len(runtime_results)} commands):")
            for runtime in runtime_results:
                status = "✅" if runtime["exit_code"] == 0 else "❌"
                report_lines.append(f"   {status} {runtime['command']}")
                if runtime["exit_code"] != 0:
                    report_lines.append(f"      Error: {runtime['output'][:200]}...")
        
        # Artifacts
        artifacts = execution_result.get("artifacts", [])
        if artifacts:
            report_lines.append(f"\n📁 Generated Artifacts ({len(artifacts)}):")
            # Show up to 10 artifacts
            for artifact in artifacts[:10]:
                report_lines.append(f"   • {artifact}")
            if len(artifacts) > 10:
                report_lines.append(f"   ... and {len(artifacts) - 10} more")
        
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

    async def generate_application(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Generate a complete application autonomously."""
        self.logger.info("Starting autonomous application generation")
        
        # Create execution plan for application generation
        execution_plan = self._extract_execution_plan(task, context or {})
        
        # Add application generation specific steps
        if not execution_plan.get("project_structure"):
            # Auto-generate project structure based on requirements
            execution_plan["project_structure"] = self._generate_project_structure(task)
            
        # Execute the application generation
        execution_result = await self._execute_in_docker(execution_plan)
        
        # Create response
        start_time = time.time()
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
                "project_structure": execution_plan.get("project_structure", {}),
                "is_autonomous_app": True
            },
            artifacts=execution_result.get("artifacts", [])
        )
        
        return response

    async def generate_application_with_checkpoints(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Generate a complete application autonomously with checkpoint support."""
        self.logger.info("Starting autonomous application generation with checkpoint support")
        
        # Check if there are existing checkpoints for this task
        existing_checkpoints = []
        if self.orchestrator and hasattr(self.orchestrator, 'get_checkpoints_for_task'):
            existing_checkpoints = await self.orchestrator.get_checkpoints_for_task(task.id)
        
        # If we have checkpoints, we could potentially resume from them
        # For now, we'll just log that we found them
        if existing_checkpoints:
            self.logger.info(f"Found {len(existing_checkpoints)} existing checkpoints for task {task.id}")
            # In a more advanced implementation, we could resume from the last successful checkpoint
        
        # Create execution plan for application generation
        execution_plan = self._extract_execution_plan(task, context or {})
        
        # Add application generation specific steps
        if not execution_plan.get("project_structure"):
            # Auto-generate project structure based on requirements
            execution_plan["project_structure"] = self._generate_project_structure(task)
            
        # Execute the application generation
        execution_result = await self._execute_in_docker(execution_plan)
        
        # Create response
        start_time = time.time()
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
                "project_structure": execution_plan.get("project_structure", {}),
                "is_autonomous_app": True,
                "checkpoints_found": len(existing_checkpoints)
            },
            artifacts=execution_result.get("artifacts", [])
        )
        
        return response

    def _generate_project_structure(self, task: Task) -> Dict[str, Any]:
        """Auto-generate project structure based on task requirements."""
        # This is a simplified implementation - in a full system, this would be more sophisticated
        project_structure = {
            "README.md": f"# {task.title}\n\n{task.description}\n",
            "src/": {},
            "tests/": {},
            ".gitignore": "__pycache__\n*.pyc\n.env\n",
        }
        
        # Add language-specific files
        if "python" in task.description.lower() or "flask" in task.description.lower():
            project_structure["requirements.txt"] = ""
            project_structure["src/main.py"] = "# Main application entry point\n"
            project_structure["src/__init__.py"] = ""
            project_structure["tests/test_main.py"] = "# Test file\n"
        elif "node" in task.description.lower() or "express" in task.description.lower():
            project_structure["package.json"] = '{"name": "auto-generated-app", "version": "1.0.0"}'
            project_structure["src/index.js"] = "// Main application entry point\n"
            project_structure["tests/test.js"] = "// Test file\n"
            
        return project_structure