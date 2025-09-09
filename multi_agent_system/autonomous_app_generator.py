"""Autonomous application generation workflow for the coding agent."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .data_structures import Task, Response, AgentType, TaskType, TaskPriority, Checkpoint, CheckpointType, AutonomousAppCheckpointData
from .enhanced_orchestrator import EnhancedMultiAgentOrchestrator
from .project_scaffolder import project_scaffolder, file_generator
from .package_manager import package_manager, dependency_analyzer
from .test_framework import test_generator, test_runner
from .output_validator import output_validator, code_formatter
from .debugger import code_debugger, troubleshooter
from .logging_config import SystemLogger


@dataclass
class AutonomousAppResult:
    """Result of autonomous application generation."""
    success: bool
    app_structure: Dict[str, Any]
    generated_files: List[str]
    dependencies: List[str]
    test_files: Dict[str, str]
    quality_metrics: Dict[str, float]
    execution_result: Optional[Response]
    issues: List[str]
    suggestions: List[str]
    confidence: float
    checkpoints: List['Checkpoint'] = None
    
    def __post_init__(self):
        if self.checkpoints is None:
            self.checkpoints = []


class AutonomousAppGenerator:
    """Generates complete applications autonomously."""
    
    def __init__(self):
        self.logger = SystemLogger("autonomous_app_generator")
        self.orchestrator = EnhancedMultiAgentOrchestrator()
        self.memory_system = None
        
    async def initialize(self) -> bool:
        """Initialize the autonomous app generator."""
        success = await self.orchestrator.initialize()
        if success:
            # Get memory system from orchestrator
            self.memory_system = self.orchestrator.memory_system
        return success
    
    async def generate_application(self, task: Task) -> AutonomousAppResult:
        """Generate a complete application autonomously."""
        start_time = time.time()
        self.logger.info("Starting autonomous application generation", task_id=task.id)
        
        # Initialize checkpoints list
        checkpoints = []
        
        try:
            # Step 1: Analyze requirements and determine project type
            project_info = await self._analyze_requirements(task)
            
            # Create checkpoint for requirements analysis
            requirements_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.REQUIREMENTS_ANALYSIS,
                data=AutonomousAppCheckpointData(project_info=project_info).dict(),
                execution_time=time.time() - start_time
            )
            requirements_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(requirements_checkpoint)
            
            # Store checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(requirements_checkpoint)
            
            # Step 2: Create project structure
            app_structure = project_scaffolder.create_project_structure(
                project_info["project_type"],
                task.title,
                project_info["dependencies"],
                project_info["language"]
            )
            
            # Create checkpoint for project structure
            structure_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.PROJECT_STRUCTURE,
                data=AutonomousAppCheckpointData(
                    project_info=project_info,
                    app_structure=app_structure
                ).dict(),
                execution_time=time.time() - start_time
            )
            structure_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(structure_checkpoint)
            
            # Store checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(structure_checkpoint)
            
            # Step 3: Generate main application code
            main_code = await self._generate_main_code(task, project_info)
            
            # Create checkpoint for main code generation
            main_code_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.MAIN_CODE_GENERATION,
                data=AutonomousAppCheckpointData(
                    project_info=project_info,
                    app_structure=app_structure,
                    main_code=main_code
                ).dict(),
                execution_time=time.time() - start_time
            )
            main_code_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(main_code_checkpoint)
            
            # Store checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(main_code_checkpoint)
            
            # Step 4: Add main code to project structure
            if project_info["language"].lower() == "python":
                app_structure["src/"]["main.py"] = main_code
            elif project_info["language"].lower() in ["javascript", "typescript"]:
                app_structure["src/"]["index.js" if project_info["language"].lower() == "javascript" else "index.ts"] = main_code
            elif project_info["language"].lower() == "go":
                app_structure["cmd/"]["main.go"] = main_code
            
            # Step 5: Generate additional files (Dockerfile, README, etc.)
            additional_files = await self._generate_additional_files(task, project_info)
            app_structure.update(additional_files)
            
            # Create checkpoint for additional files
            additional_files_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.ADDITIONAL_FILES,
                data=AutonomousAppCheckpointData(
                    project_info=project_info,
                    app_structure=app_structure,
                    additional_files=additional_files
                ).dict(),
                execution_time=time.time() - start_time
            )
            additional_files_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(additional_files_checkpoint)
            
            # Store checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(additional_files_checkpoint)
            
            # Step 6: Generate tests
            test_files = test_generator.generate_tests(
                main_code, 
                project_info["language"], 
                project_info["test_framework"]
            )
            app_structure.update(test_files)
            
            # Create checkpoint for test generation
            test_generation_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.TEST_GENERATION,
                data=AutonomousAppCheckpointData(
                    project_info=project_info,
                    app_structure=app_structure,
                    test_files=test_files
                ).dict(),
                execution_time=time.time() - start_time
            )
            test_generation_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(test_generation_checkpoint)
            
            # Store checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(test_generation_checkpoint)
            
            # Step 7: Validate code quality
            validation_result = output_validator.validate_output(
                main_code, 
                project_info["language"], 
                task.requirements
            )
            
            # Create checkpoint for code validation
            validation_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.CODE_VALIDATION,
                data=AutonomousAppCheckpointData(
                    project_info=project_info,
                    app_structure=app_structure,
                    test_files=test_files,
                    validation_result=validation_result.__dict__ if hasattr(validation_result, '__dict__') else validation_result
                ).dict(),
                execution_time=time.time() - start_time
            )
            validation_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(validation_checkpoint)
            
            # Store checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(validation_checkpoint)
            
            # Step 8: Format code
            formatted_code = code_formatter.format_code(main_code, project_info["language"])
            if project_info["language"].lower() == "python":
                app_structure["src/"]["main.py"] = formatted_code
            elif project_info["language"].lower() in ["javascript", "typescript"]:
                app_structure["src/"]["index.js" if project_info["language"].lower() == "javascript" else "index.ts"] = formatted_code
            elif project_info["language"].lower() == "go":
                app_structure["cmd/"]["main.go"] = formatted_code
            
            # Create checkpoint for code formatting
            formatting_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.CODE_FORMATTING,
                data=AutonomousAppCheckpointData(
                    project_info=project_info,
                    app_structure=app_structure,
                    formatted_code=formatted_code
                ).dict(),
                execution_time=time.time() - start_time
            )
            formatting_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(formatting_checkpoint)
            
            # Store checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(formatting_checkpoint)
            
            # Step 9: Execute the application in a test environment
            execution_result = await self._execute_application(task, app_structure, project_info)
            
            # Create checkpoint for application execution
            execution_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.APPLICATION_EXECUTION,
                data=AutonomousAppCheckpointData(
                    project_info=project_info,
                    app_structure=app_structure,
                    execution_result=execution_result.__dict__ if hasattr(execution_result, '__dict__') else execution_result
                ).dict(),
                execution_time=time.time() - start_time
            )
            execution_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(execution_checkpoint)
            
            # Store checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(execution_checkpoint)
            
            # Step 10: Compile results
            generated_files = list(app_structure.keys())
            
            result = AutonomousAppResult(
                success=execution_result.success if execution_result else True,
                app_structure=app_structure,
                generated_files=generated_files,
                dependencies=project_info["dependencies"],
                test_files=test_files,
                quality_metrics={
                    "readability": validation_result.quality_metrics.readability_score,
                    "maintainability": validation_result.quality_metrics.maintainability_score,
                    "efficiency": validation_result.quality_metrics.efficiency_score,
                    "security": validation_result.quality_metrics.security_score,
                    "testability": validation_result.quality_metrics.testability_score,
                    "overall": validation_result.quality_metrics.overall_score
                },
                execution_result=execution_result,
                issues=validation_result.issues,
                suggestions=validation_result.suggestions,
                confidence=validation_result.confidence,
                checkpoints=checkpoints
            )
            
            # Create final checkpoint
            final_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.FINAL_RESULT,
                data=AutonomousAppCheckpointData(
                    project_info=project_info,
                    app_structure=app_structure,
                    generated_files=generated_files,
                    dependencies=project_info["dependencies"],
                    test_files=test_files,
                    quality_metrics={
                        "readability": validation_result.quality_metrics.readability_score,
                        "maintainability": validation_result.quality_metrics.maintainability_score,
                        "efficiency": validation_result.quality_metrics.efficiency_score,
                        "security": validation_result.quality_metrics.security_score,
                        "testability": validation_result.quality_metrics.testability_score,
                        "overall": validation_result.quality_metrics.overall_score
                    },
                    issues=validation_result.issues,
                    suggestions=validation_result.suggestions,
                    confidence=validation_result.confidence
                ).dict(),
                execution_time=time.time() - start_time
            )
            final_checkpoint.mark_completed(time.time() - start_time)
            checkpoints.append(final_checkpoint)
            
            # Store final checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(final_checkpoint)
            
            self.logger.info("Autonomous application generation completed", 
                           task_id=task.id, 
                           execution_time=time.time() - start_time,
                           success=result.success)
            
            return result
            
        except Exception as e:
            self.logger.error("Autonomous application generation failed", 
                            task_id=task.id, 
                            error=str(e))
            
            # Create error checkpoint
            error_checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=CheckpointType.FINAL_RESULT,
                data=AutonomousAppCheckpointData(
                    issues=[str(e)],
                    suggestions=["Review the error and try again with more specific requirements"]
                ).dict(),
                execution_time=time.time() - start_time
            )
            error_checkpoint.mark_failed(str(e), time.time() - start_time)
            checkpoints.append(error_checkpoint)
            
            # Store error checkpoint in memory system if available
            if self.memory_system:
                await self.memory_system.store_checkpoint(error_checkpoint)
            
            return AutonomousAppResult(
                success=False,
                app_structure={},
                generated_files=[],
                dependencies=[],
                test_files={},
                quality_metrics={
                    "readability": 0.0,
                    "maintainability": 0.0,
                    "efficiency": 0.0,
                    "security": 0.0,
                    "testability": 0.0,
                    "overall": 0.0
                },
                execution_result=None,
                issues=[str(e)],
                suggestions=["Review the error and try again with more specific requirements"],
                confidence=0.0,
                checkpoints=checkpoints
            )
    
    async def _analyze_requirements(self, task: Task) -> Dict[str, Any]:
        """Analyze task requirements to determine project characteristics."""
        self.logger.info("Analyzing requirements", task_id=task.id)
        
        # Determine language
        language = "python"  # Default
        task_content = f"{task.title} {task.description} {' '.join(task.requirements)}".lower()
        
        if "javascript" in task_content or "node" in task_content:
            language = "javascript"
        elif "typescript" in task_content:
            language = "typescript"
        elif "go" in task_content or "golang" in task_content:
            language = "go"
        elif "rust" in task_content:
            language = "rust"
        elif "java" in task_content:
            language = "java"
        
        # Determine project type
        project_type = "generic"
        if "web" in task_content and "api" in task_content:
            project_type = "python-web-api" if language == "python" else "javascript-api"
        elif "web" in task_content:
            project_type = "javascript-web-app" if language in ["javascript", "typescript"] else "python-web-api"
        elif "data" in task_content and ("analysis" in task_content or "science" in task_content):
            project_type = "python-data-analysis"
        elif "cli" in task_content or "command" in task_content:
            project_type = "python-cli"
        
        # Resolve dependencies
        dependencies = package_manager.resolve_dependencies(task.description, language)
        
        # Determine test framework
        test_framework = "auto"
        
        return {
            "language": language,
            "project_type": project_type,
            "dependencies": dependencies,
            "test_framework": test_framework
        }
    
    async def _generate_main_code(self, task: Task, project_info: Dict[str, Any]) -> str:
        """Generate the main application code."""
        self.logger.info("Generating main application code", task_id=task.id)
        
        # Create a coding task for the coder agent
        coding_task = Task(
            type=TaskType.CODING,
            priority=TaskPriority.HIGH,
            title=f"Generate {project_info['project_type']} application",
            description=task.description,
            requirements=task.requirements,
            context={
                "language": project_info["language"],
                "project_type": project_info["project_type"],
                "dependencies": project_info["dependencies"]
            }
        )
        
        # Use the orchestrator's coder agent to generate code
        coder_agent = self.orchestrator.agents[AgentType.CODER]
        response = await coder_agent.process_task(coding_task)
        
        if response.success:
            return response.content
        else:
            # Fallback: generate basic template
            return self._generate_fallback_code(project_info)
    
    def _generate_fallback_code(self, project_info: Dict[str, Any]) -> str:
        """Generate fallback code when agent fails."""
        language = project_info["language"].lower()
        
        if language == "python":
            if "web-api" in project_info["project_type"]:
                return '''"""
Auto-generated Web API Application
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello from auto-generated API!"})

@app.route('/api/status')
def status():
    return jsonify({"status": "running", "service": "auto-generated"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
            elif "data-analysis" in project_info["project_type"]:
                return '''"""
Auto-generated Data Analysis Application
"""

import pandas as pd

def analyze_data():
    """Perform basic data analysis"""
    # Placeholder for data analysis logic
    print("Performing data analysis...")
    return {"status": "analysis_complete"}

if __name__ == '__main__':
    result = analyze_data()
    print(result)
'''
            else:
                return '''"""
Auto-generated Python Application
"""

def main():
    print("Hello from auto-generated Python application!")

if __name__ == '__main__':
    main()
'''
        elif language in ["javascript", "typescript"]:
            return '''/**
 * Auto-generated JavaScript/TypeScript Application
 */

function main() {
    console.log("Hello from auto-generated application!");
}

main();
'''
        elif language == "go":
            return '''// Auto-generated Go Application
package main

import "fmt"

func main() {
    fmt.Println("Hello from auto-generated Go application!")
}
'''
        else:
            return f'// Auto-generated {language} Application\n// TODO: Add implementation'
    
    async def _generate_additional_files(self, task: Task, project_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate additional project files."""
        self.logger.info("Generating additional project files", task_id=task.id)
        
        additional_files = {}
        
        # Generate Dockerfile
        dockerfile = file_generator.generate_file(
            "dockerfile",
            language=project_info["language"],
            app_name=task.title.lower().replace(" ", "_")
        )
        additional_files["Dockerfile"] = dockerfile
        
        # Generate README
        readme = file_generator.generate_file(
            "readme",
            project_name=task.title,
            description=task.description
        )
        additional_files["README.md"] = readme
        
        # Generate .gitignore
        gitignore = file_generator.generate_file(
            "gitignore",
            language=project_info["language"]
        )
        additional_files[".gitignore"] = gitignore
        
        return additional_files
    
    async def _execute_application(self, task: Task, app_structure: Dict[str, Any], 
                                 project_info: Dict[str, Any]) -> Optional[Response]:
        """Execute the generated application in a test environment."""
        self.logger.info("Executing generated application", task_id=task.id)
        
        try:
            # Create an execution task
            execution_task = Task(
                type=TaskType.EXECUTION,
                priority=TaskPriority.MEDIUM,
                title=f"Execute {task.title}",
                description=f"Run the auto-generated {project_info['project_type']} application",
                requirements=["Execute the generated application and verify it works"],
                context={
                    "code_files": self._flatten_structure(app_structure),
                    "language": project_info["language"],
                    "entry_point": self._get_entry_point(project_info["language"]),
                    "requirements": project_info["dependencies"],
                    "tests": {},  # Tests are handled separately
                    "timeout": 120  # 2 minutes timeout
                }
            )
            
            # Use the orchestrator's execution agent
            execution_agent = self.orchestrator.agents[AgentType.EXECUTION]
            response = await execution_agent.process_task(execution_task)
            
            return response
            
        except Exception as e:
            self.logger.error("Application execution failed", task_id=task.id, error=str(e))
            return None
    
    def _flatten_structure(self, structure: Dict[str, Any], base_path: str = "") -> Dict[str, str]:
        """Flatten nested directory structure into file paths and contents."""
        files = {}
        
        for path, content in structure.items():
            full_path = f"{base_path}{path}" if base_path else path
            
            if isinstance(content, dict):
                # Recursively flatten subdirectories
                sub_files = self._flatten_structure(content, f"{full_path}")
                files.update(sub_files)
            else:
                # Add file
                files[full_path] = content if isinstance(content, str) else str(content)
                
        return files
    
    def _get_entry_point(self, language: str) -> str:
        """Get the entry point file for the language."""
        entry_points = {
            "python": "src/main.py",
            "javascript": "src/index.js",
            "typescript": "src/index.ts",
            "go": "cmd/main.go"
        }
        return entry_points.get(language.lower(), "src/main.py")
    
    async def shutdown(self) -> None:
        """Shutdown the autonomous app generator."""
        await self.orchestrator.shutdown()


# Convenience function for easy use
async def generate_autonomous_application(task_description: str, 
                                        title: str = "Autonomous Application") -> AutonomousAppResult:
    """Convenience function to generate an autonomous application."""
    generator = AutonomousAppGenerator()
    
    # Initialize
    if not await generator.initialize():
        return AutonomousAppResult(
            success=False,
            app_structure={},
            generated_files=[],
            dependencies=[],
            test_files={},
            quality_metrics={
                "readability": 0.0,
                "maintainability": 0.0,
                "efficiency": 0.0,
                "security": 0.0,
                "testability": 0.0,
                "overall": 0.0
            },
            execution_result=None,
            issues=["Failed to initialize generator"],
            suggestions=["Check system requirements and dependencies"],
            confidence=0.0,
            checkpoints=[]
        )
    
    # Create task
    task = Task(
        type=TaskType.EXECUTION,
        priority=TaskPriority.HIGH,
        title=title,
        description=task_description,
        requirements=[task_description]
    )
    
    # Generate application
    result = await generator.generate_application(task)
    
    # Shutdown
    await generator.shutdown()
    
    return result