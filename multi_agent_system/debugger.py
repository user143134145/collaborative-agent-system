"""Debugging and troubleshooting system for autonomous coding agent."""

import re
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class DebugInfo:
    """Information about debugging results."""
    issues_found: List[str]
    suggestions: List[str]
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 to 1.0


class CodeDebugger:
    """Analyzes code for common issues and provides debugging suggestions."""
    
    def __init__(self):
        self.common_issues = {
            "python": self._debug_python_code,
            "javascript": self._debug_javascript_code,
            "typescript": self._debug_javascript_code,
            "go": self._debug_go_code
        }
    
    def debug_code(self, code: str, language: str = "python", error_output: str = "") -> DebugInfo:
        """Debug code and return debugging information."""
        if language.lower() in self.common_issues:
            debugger = self.common_issues[language.lower()]
            return debugger(code, error_output)
        else:
            return self._debug_generic_code(code, error_output)
    
    def _debug_python_code(self, code: str, error_output: str = "") -> DebugInfo:
        """Debug Python code for common issues."""
        issues = []
        suggestions = []
        
        # Check for common Python syntax issues
        if "IndentationError" in error_output or re.search(r"indentation|expected an indented block", code):
            issues.append("Indentation error detected")
            suggestions.append("Check that all code blocks are properly indented with consistent spacing (use either spaces or tabs, not both)")
        
        if "SyntaxError" in error_output:
            issues.append("Syntax error detected")
            suggestions.append("Check for missing colons, parentheses, brackets, or quotes")
        
        if "NameError" in error_output:
            issues.append("Name error detected")
            suggestions.append("Check that all variables are defined before use and imported modules are correctly imported")
        
        if "ImportError" in error_output or "ModuleNotFoundError" in error_output:
            issues.append("Import error detected")
            suggestions.append("Check that all required modules are installed and correctly imported")
        
        # Check for common code patterns that might cause issues
        if "print(" in code and "import" not in code:
            issues.append("Potential missing imports")
            suggestions.append("Consider adding necessary import statements")
        
        if re.search(r"open\([^)]*\)", code) and "with" not in code:
            issues.append("File handling without context manager")
            suggestions.append("Use 'with' statement for proper file handling to ensure files are closed")
        
        # Check for error output patterns
        if "division by zero" in error_output.lower():
            issues.append("Division by zero error")
            suggestions.append("Add checks to prevent division by zero")
        
        if "index out of range" in error_output.lower():
            issues.append("Index out of range error")
            suggestions.append("Add bounds checking before accessing list/tuple elements")
        
        if "key error" in error_output.lower():
            issues.append("Key error")
            suggestions.append("Check that dictionary keys exist before accessing them, or use .get() method")
        
        # Analyze code for potential issues
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("if") and ":" not in line:
                issues.append(f"Possible syntax error on line {i}")
                suggestions.append(f"Add colon ':' at the end of line {i}")
            
            if re.search(r"=\s*=\s*", line):
                issues.append(f"Possible assignment error on line {i}")
                suggestions.append(f"Use single '=' for assignment, '==' for comparison on line {i}")
        
        # Determine severity
        severity = "low"
        confidence = 0.8
        if issues:
            if any("critical" in issue.lower() for issue in issues):
                severity = "critical"
            elif any("error" in issue.lower() for issue in issues):
                severity = "high"
            elif len(issues) > 2:
                severity = "medium"
        
        return DebugInfo(
            issues_found=issues,
            suggestions=suggestions,
            severity=severity,
            confidence=confidence
        )
    
    def _debug_javascript_code(self, code: str, error_output: str = "") -> DebugInfo:
        """Debug JavaScript/TypeScript code for common issues."""
        issues = []
        suggestions = []
        
        # Check for common JavaScript syntax issues
        if "SyntaxError" in error_output:
            issues.append("Syntax error detected")
            suggestions.append("Check for missing semicolons, braces, parentheses, or quotes")
        
        if "ReferenceError" in error_output:
            issues.append("Reference error detected")
            suggestions.append("Check that all variables are declared before use")
        
        if "TypeError" in error_output:
            issues.append("Type error detected")
            suggestions.append("Check that variables are of the expected type before operations")
        
        # Check for common code patterns
        if re.search(r"\.forEach\(|\.map\(", code) and "=>" not in code and "function" not in code:
            issues.append("Arrow function syntax issue")
            suggestions.append("Use arrow function syntax (=>) or function keyword in array methods")
        
        if "console.log(" in code and "import" not in code and "require" not in code:
            issues.append("Potential missing imports")
            suggestions.append("Consider adding necessary import/require statements")
        
        # Check for error output patterns
        if "undefined is not a function" in error_output.lower():
            issues.append("Function not defined")
            suggestions.append("Check that functions are properly defined before calling them")
        
        if "cannot read property" in error_output.lower():
            issues.append("Property access on undefined")
            suggestions.append("Check that objects are properly initialized before accessing their properties")
        
        # Analyze code for potential issues
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if line.strip().endswith("{") and not line.strip().startswith("//"):
                # Check if next line is properly indented
                if i < len(lines) and lines[i].strip() and not lines[i].startswith(" "):
                    issues.append(f"Possible indentation issue after line {i}")
                    suggestions.append(f"Ensure proper indentation for code blocks starting at line {i}")
        
        # Determine severity
        severity = "low"
        confidence = 0.8
        if issues:
            if any("critical" in issue.lower() for issue in issues):
                severity = "critical"
            elif any("error" in issue.lower() for issue in issues):
                severity = "high"
            elif len(issues) > 2:
                severity = "medium"
        
        return DebugInfo(
            issues_found=issues,
            suggestions=suggestions,
            severity=severity,
            confidence=confidence
        )
    
    def _debug_go_code(self, code: str, error_output: str = "") -> DebugInfo:
        """Debug Go code for common issues."""
        issues = []
        suggestions = []
        
        # Check for common Go syntax issues
        if "syntax error" in error_output.lower():
            issues.append("Syntax error detected")
            suggestions.append("Check for missing braces, parentheses, or semicolons")
        
        if "undefined:" in error_output:
            issues.append("Undefined identifier")
            suggestions.append("Check that all variables and functions are properly declared")
        
        if "imported and not used" in error_output:
            issues.append("Unused import")
            suggestions.append("Remove unused import statements or use the imported package")
        
        # Check for common code patterns
        if "func main()" in code and "package main" not in code:
            issues.append("Missing package declaration")
            suggestions.append("Add 'package main' at the beginning of the file")
        
        # Check for error output patterns
        if "cannot use" in error_output and "as type" in error_output:
            issues.append("Type mismatch")
            suggestions.append("Check that variables are of the correct type for the operation")
        
        # Analyze code for potential issues
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if "if" in line and "{" not in line:
                issues.append(f"Possible syntax error on line {i}")
                suggestions.append(f"Add opening brace '{{' on line {i} or at the end of the if statement")
        
        # Determine severity
        severity = "low"
        confidence = 0.8
        if issues:
            if any("critical" in issue.lower() for issue in issues):
                severity = "critical"
            elif any("error" in issue.lower() for issue in issues):
                severity = "high"
            elif len(issues) > 2:
                severity = "medium"
        
        return DebugInfo(
            issues_found=issues,
            suggestions=suggestions,
            severity=severity,
            confidence=confidence
        )
    
    def _debug_generic_code(self, code: str, error_output: str = "") -> DebugInfo:
        """Debug generic code for common issues."""
        issues = []
        suggestions = []
        
        # Generic error pattern matching
        if "error" in error_output.lower():
            issues.append("Error detected in execution")
            suggestions.append("Review the error output and check the corresponding code")
        
        if "exception" in error_output.lower():
            issues.append("Exception occurred")
            suggestions.append("Add proper exception handling")
        
        # Basic syntax checks
        if code.count("(") != code.count(")"):
            issues.append("Mismatched parentheses")
            suggestions.append("Check that all opening parentheses have corresponding closing parentheses")
        
        if code.count("{") != code.count("}"):
            issues.append("Mismatched braces")
            suggestions.append("Check that all opening braces have corresponding closing braces")
        
        if code.count("[") != code.count("]"):
            issues.append("Mismatched brackets")
            suggestions.append("Check that all opening brackets have corresponding closing brackets")
        
        # Determine severity
        severity = "low"
        confidence = 0.6
        if issues:
            if any("critical" in issue.lower() for issue in issues):
                severity = "critical"
            elif any("error" in issue.lower() for issue in issues):
                severity = "high"
            elif len(issues) > 2:
                severity = "medium"
        
        return DebugInfo(
            issues_found=issues,
            suggestions=suggestions,
            severity=severity,
            confidence=confidence
        )


class Troubleshooter:
    """Provides troubleshooting solutions for common problems."""
    
    def __init__(self):
        self.solutions = {
            "dependency_installation": self._solve_dependency_issues,
            "runtime_errors": self._solve_runtime_errors,
            "test_failures": self._solve_test_failures,
            "docker_issues": self._solve_docker_issues
        }
    
    def solve_problem(self, problem_type: str, context: Dict[str, Any]) -> List[str]:
        """Provide solutions for a specific type of problem."""
        if problem_type in self.solutions:
            solver = self.solutions[problem_type]
            return solver(context)
        else:
            return [f"No specific solution for {problem_type}. General troubleshooting steps:", 
                   "1. Check error messages carefully", 
                   "2. Verify all dependencies are installed", 
                   "3. Ensure proper configuration", 
                   "4. Consult documentation"]
    
    def _solve_dependency_issues(self, context: Dict[str, Any]) -> List[str]:
        """Solve dependency installation issues."""
        solutions = [
            "Dependency Installation Issues:",
            "1. Check internet connection and package repository availability",
            "2. Verify package names are correct and available in the package index",
            "3. Try updating package manager: pip install --upgrade pip (for Python)",
            "4. Clear package manager cache: pip cache purge (for Python)",
            "5. Check for conflicting package versions",
            "6. Try installing packages one by one to identify problematic packages",
            "7. Consider using virtual environments to isolate dependencies"
        ]
        
        # Add language-specific solutions
        language = context.get("language", "python").lower()
        if language == "python":
            solutions.extend([
                "8. For Python: Try 'pip install --no-cache-dir <package>'",
                "9. For Python: Check requirements.txt for version conflicts",
                "10. For Python: Consider using conda instead of pip for scientific packages"
            ])
        elif language in ["javascript", "typescript"]:
            solutions.extend([
                "8. For JavaScript/TypeScript: Try 'npm install --force'",
                "9. For JavaScript/TypeScript: Clear node_modules and package-lock.json",
                "10. For JavaScript/TypeScript: Check Node.js version compatibility"
            ])
        
        return solutions
    
    def _solve_runtime_errors(self, context: Dict[str, Any]) -> List[str]:
        """Solve runtime errors."""
        solutions = [
            "Runtime Error Troubleshooting:",
            "1. Check error traceback to identify exact line and cause",
            "2. Verify all variables are properly initialized before use",
            "3. Check file paths and ensure files exist",
            "4. Verify environment variables are set correctly",
            "5. Check resource limits (memory, disk space, etc.)",
            "6. Add logging to trace execution flow",
            "7. Use debugging tools to step through code",
            "8. Check for race conditions in concurrent code"
        ]
        
        error_type = context.get("error_type", "").lower()
        if "permission" in error_type:
            solutions.extend([
                "9. For permission errors: Check file permissions and ownership",
                "10. For permission errors: Run with appropriate privileges or change file permissions"
            ])
        elif "memory" in error_type:
            solutions.extend([
                "9. For memory errors: Optimize memory usage or increase available memory",
                "10. For memory errors: Check for memory leaks in the code"
            ])
        
        return solutions
    
    def _solve_test_failures(self, context: Dict[str, Any]) -> List[str]:
        """Solve test failures."""
        solutions = [
            "Test Failure Troubleshooting:",
            "1. Run tests in verbose mode to get detailed output",
            "2. Check test assertions and expected vs actual values",
            "3. Verify test data and setup/teardown procedures",
            "4. Check for test isolation issues (shared state between tests)",
            "5. Ensure test environment matches development environment",
            "6. Verify all test dependencies are properly installed",
            "7. Check for timing issues in asynchronous tests",
            "8. Review test coverage to ensure all code paths are tested"
        ]
        
        return solutions
    
    def _solve_docker_issues(self, context: Dict[str, Any]) -> List[str]:
        """Solve Docker-related issues."""
        solutions = [
            "Docker Troubleshooting:",
            "1. Check if Docker daemon is running: systemctl status docker (Linux)",
            "2. Verify Docker installation and version: docker --version",
            "3. Check available disk space: df -h",
            "4. Verify Docker permissions: Add user to docker group",
            "5. Check Docker image availability: docker images",
            "6. Verify container resource limits: docker stats",
            "7. Check network connectivity from container: docker run --rm busybox ping -c 3 google.com",
            "8. Review Dockerfile for syntax errors: docker build --no-cache"
        ]
        
        error_message = context.get("error_message", "").lower()
        if "permission denied" in error_message:
            solutions.extend([
                "9. For permission issues: Run Docker with sudo or add user to docker group",
                "10. For permission issues: Check Docker socket permissions: ls -la /var/run/docker.sock"
            ])
        elif "no space left" in error_message:
            solutions.extend([
                "9. For disk space issues: Clean up unused Docker images: docker system prune -a",
                "10. For disk space issues: Remove dangling images: docker image prune"
            ])
        
        return solutions


# Global instances
code_debugger = CodeDebugger()
troubleshooter = Troubleshooter()