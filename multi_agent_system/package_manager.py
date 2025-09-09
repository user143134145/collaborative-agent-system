"""Package management system for autonomous coding agent."""

import json
import re
from typing import Dict, List, Set, Optional, Any
from pathlib import Path


class PackageManager:
    """Manages package dependencies for different programming languages."""
    
    def __init__(self):
        self.supported_languages = {
            "python": self._resolve_python_dependencies,
            "javascript": self._resolve_javascript_dependencies,
            "typescript": self._resolve_javascript_dependencies,
            "go": self._resolve_go_dependencies,
            "java": self._resolve_java_dependencies,
            "rust": self._resolve_rust_dependencies
        }
        
        # Common package mappings for different tasks
        self.task_package_mappings = {
            "web scraping": ["requests", "beautifulsoup4", "lxml"],
            "data analysis": ["pandas", "numpy", "matplotlib"],
            "machine learning": ["scikit-learn", "tensorflow", "torch"],
            "web development": ["flask", "django", "fastapi"],
            "api development": ["fastapi", "uvicorn", "pydantic"],
            "database": ["sqlalchemy", "psycopg2", "pymongo"],
            "testing": ["pytest", "unittest", "coverage"],
            "image processing": ["pillow", "opencv-python", "scikit-image"],
            "natural language processing": ["nltk", "spacy", "transformers"],
        }
    
    def resolve_dependencies(self, task_description: str, language: str = "python") -> List[str]:
        """Resolve dependencies based on task description and language."""
        if language.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        resolver = self.supported_languages[language.lower()]
        return resolver(task_description)
    
    def _resolve_python_dependencies(self, task_description: str) -> List[str]:
        """Resolve Python dependencies based on task description."""
        dependencies = set()
        
        # Check for task-specific packages
        task_lower = task_description.lower()
        for task_type, packages in self.task_package_mappings.items():
            if task_type in task_lower:
                dependencies.update(packages)
        
        # Check for specific keywords in the task description
        if "web" in task_lower and "scrap" in task_lower:
            dependencies.update(["requests", "beautifulsoup4"])
        elif "web" in task_lower and "api" in task_lower:
            dependencies.update(["fastapi", "uvicorn"])
        elif "web" in task_lower:
            dependencies.update(["flask"])
            
        if "data" in task_lower:
            dependencies.update(["pandas", "numpy"])
            
        if "ml" in task_lower or "machine learning" in task_lower:
            dependencies.update(["scikit-learn"])
            
        if "test" in task_lower:
            dependencies.update(["pytest"])
            
        if "image" in task_lower:
            dependencies.update(["pillow"])
            
        if "nlp" in task_lower or "natural language" in task_lower:
            dependencies.update(["nltk"])
            
        # Always include basic packages
        dependencies.update(["pytest"])  # For testing
        
        return list(dependencies)
    
    def _resolve_javascript_dependencies(self, task_description: str) -> List[str]:
        """Resolve JavaScript/TypeScript dependencies based on task description."""
        dependencies = set()
        
        task_lower = task_description.lower()
        
        # Web development frameworks
        if "web" in task_lower and "framework" in task_lower:
            dependencies.update(["express"])
        elif "web" in task_lower:
            dependencies.update(["express"])
            
        # React for frontend
        if "react" in task_lower:
            dependencies.update(["react", "react-dom"])
            
        # Database
        if "database" in task_lower or "mongo" in task_lower:
            dependencies.update(["mongoose"])
        elif "sql" in task_lower:
            dependencies.update(["sequelize"])
            
        # Testing
        if "test" in task_lower:
            dependencies.update(["jest", "@types/jest"])
            
        # Utilities
        dependencies.update(["lodash"])
        
        return list(dependencies)
    
    def _resolve_go_dependencies(self, task_description: str) -> List[str]:
        """Resolve Go dependencies based on task description."""
        dependencies = set()
        
        task_lower = task_description.lower()
        
        # Web framework
        if "web" in task_lower:
            dependencies.update(["github.com/gin-gonic/gin"])
            
        # Database
        if "database" in task_lower or "sql" in task_lower:
            dependencies.update(["github.com/jinzhu/gorm"])
            
        # Testing
        if "test" in task_lower:
            dependencies.update(["github.com/stretchr/testify"])
            
        return list(dependencies)
    
    def _resolve_java_dependencies(self, task_description: str) -> List[str]:
        """Resolve Java dependencies based on task description."""
        dependencies = set()
        
        task_lower = task_description.lower()
        
        # Spring framework for web apps
        if "web" in task_lower:
            dependencies.update(["org.springframework:spring-web"])
            
        # Database
        if "database" in task_lower:
            dependencies.update(["org.springframework:spring-jdbc"])
            
        # Testing
        if "test" in task_lower:
            dependencies.update(["junit:junit:4.13.2"])
            
        return list(dependencies)
    
    def _resolve_rust_dependencies(self, task_description: str) -> List[str]:
        """Resolve Rust dependencies based on task description."""
        dependencies = set()
        
        task_lower = task_description.lower()
        
        # Web framework
        if "web" in task_lower:
            dependencies.update(["actix-web = \"4.0\""])
            
        # Serialization
        if "json" in task_lower or "serialize" in task_lower:
            dependencies.update(["serde = { version = \"1.0\", features = [\"derive\"] }"])
            
        # Testing
        if "test" in task_lower:
            dependencies.update(["tokio = { version = \"1.0\", features = [\"rt\", \"rt-multi-thread\", \"macros\"] }"])
            
        return list(dependencies)
    
    def generate_requirements_file(self, dependencies: List[str], language: str = "python") -> str:
        """Generate a requirements file content for the specified language."""
        if language.lower() == "python":
            return "\n".join(dependencies)
        elif language.lower() in ["javascript", "typescript"]:
            package_json = {
                "name": "autonomous-app",
                "version": "1.0.0",
                "dependencies": {},
                "devDependencies": {}
            }
            
            # For simplicity, we're putting all dependencies in regular dependencies
            for dep in dependencies:
                # Simple parsing - in a real implementation, we'd want to handle versions
                package_name = dep.split("@")[0] if "@" in dep else dep
                package_json["dependencies"][package_name] = "latest"
                
            return json.dumps(package_json, indent=2)
        elif language.lower() == "go":
            # For Go, we return a list of module paths
            return "\n".join(dependencies)
        else:
            return "\n".join(dependencies)
    
    def suggest_optimizations(self, current_dependencies: List[str], language: str = "python") -> List[str]:
        """Suggest optimizations for the current dependencies."""
        suggestions = []
        
        if language.lower() == "python":
            # Check for common optimization opportunities
            if "tensorflow" in current_dependencies and "keras" in current_dependencies:
                suggestions.append("Consider using tensorflow.keras instead of separate keras package")
                
            if "flask" in current_dependencies and "flask-restful" in current_dependencies:
                suggestions.append("Consider using Flask's built-in REST capabilities instead of flask-restful")
                
            if "pandas" in current_dependencies and "dask" in current_dependencies:
                suggestions.append("For large datasets, consider using only dask.dataframe instead of both pandas and dask")
        
        return suggestions


class DependencyAnalyzer:
    """Analyzes code to detect required dependencies."""
    
    def __init__(self):
        self.import_patterns = {
            "python": r"(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
            "javascript": r"(?:import|require\()\s*['\"]([a-zA-Z_][a-zA-Z0-9_.-]*)",
            "go": r"import\s+(?:\([\s\S]*?['\"]([a-zA-Z0-9./-]+)['\"][\s\S]*?\)|['\"]([a-zA-Z0-9./-]+)['\"])",
        }
    
    def analyze_code_dependencies(self, code: str, language: str = "python") -> List[str]:
        """Analyze code to detect imported packages."""
        if language.lower() not in self.import_patterns:
            return []
            
        pattern = self.import_patterns[language.lower()]
        matches = re.findall(pattern, code)
        
        dependencies = set()
        for match in matches:
            # Handle tuple matches from regex groups
            if isinstance(match, tuple):
                pkg = next((m for m in match if m), None)
            else:
                pkg = match
                
            if pkg:
                # For Python, we only want the top-level package
                if language.lower() == "python":
                    pkg = pkg.split('.')[0]
                dependencies.add(pkg)
                
        return list(dependencies)
    
    def generate_missing_dependencies(self, code_dependencies: List[str], installed_dependencies: List[str]) -> List[str]:
        """Find missing dependencies."""
        code_set = set(code_dependencies)
        installed_set = set(installed_dependencies)
        return list(code_set - installed_set)


# Global instances
package_manager = PackageManager()
dependency_analyzer = DependencyAnalyzer()