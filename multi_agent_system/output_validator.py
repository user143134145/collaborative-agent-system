"""Output validation and quality checks for autonomous coding agent."""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Metrics for code quality assessment."""
    readability_score: float  # 0.0 to 1.0
    maintainability_score: float  # 0.0 to 1.0
    efficiency_score: float  # 0.0 to 1.0
    security_score: float  # 0.0 to 1.0
    testability_score: float  # 0.0 to 1.0
    overall_score: float  # 0.0 to 1.0


@dataclass
class ValidationResult:
    """Result of output validation."""
    is_valid: bool
    issues: List[str]
    suggestions: List[str]
    quality_metrics: QualityMetrics
    confidence: float  # 0.0 to 1.0


class OutputValidator:
    """Validates and assesses quality of generated code."""
    
    def __init__(self):
        self.quality_checkers = {
            "python": self._validate_python_code,
            "javascript": self._validate_javascript_code,
            "typescript": self._validate_javascript_code,
            "go": self._validate_go_code
        }
    
    def validate_output(self, code: str, language: str = "python", 
                       requirements: List[str] = None) -> ValidationResult:
        """Validate code output and assess quality."""
        if language.lower() in self.quality_checkers:
            validator = self.quality_checkers[language.lower()]
            return validator(code, requirements or [])
        else:
            return self._validate_generic_code(code, requirements or [])
    
    def _validate_python_code(self, code: str, requirements: List[str]) -> ValidationResult:
        """Validate Python code and assess quality."""
        issues = []
        suggestions = []
        
        # Basic syntax checks
        if code.count("(") != code.count(")"):
            issues.append("Mismatched parentheses")
        if code.count("{") != code.count("}"):
            issues.append("Mismatched braces")
        if code.count("[") != code.count("]"):
            issues.append("Mismatched brackets")
        
        # Check for required elements
        missing_elements = []
        for req in requirements:
            if req.lower() not in code.lower():
                missing_elements.append(req)
        
        if missing_elements:
            issues.append(f"Missing required elements: {', '.join(missing_elements)}")
        
        # Quality metrics calculation
        readability = self._assess_readability(code)
        maintainability = self._assess_maintainability(code)
        efficiency = self._assess_efficiency(code)
        security = self._assess_security(code)
        testability = self._assess_testability(code)
        
        overall = (readability + maintainability + efficiency + security + testability) / 5
        
        # Add suggestions based on quality scores
        if readability < 0.7:
            suggestions.append("Improve code readability by using descriptive variable names and adding comments")
        if maintainability < 0.7:
            suggestions.append("Improve maintainability by breaking down large functions and reducing complexity")
        if efficiency < 0.7:
            suggestions.append("Optimize code efficiency by reviewing algorithms and data structures")
        if security < 0.7:
            suggestions.append("Enhance security by validating inputs and handling exceptions properly")
        if testability < 0.7:
            suggestions.append("Improve testability by writing modular code and adding unit tests")
        
        # Check for Python-specific best practices
        if "import" not in code:
            suggestions.append("Consider adding import statements for required modules")
        
        if "def " in code and "return" not in code:
            suggestions.append("Functions should generally return values or have clear side effects")
        
        if re.search(r"print\([^)]*\)", code) and "__name__" not in code:
            suggestions.append("Consider using proper logging instead of print statements for production code")
        
        # Validate docstrings
        if "def " in code and '"""' not in code:
            suggestions.append("Add docstrings to functions and classes for better documentation")
        
        quality_metrics = QualityMetrics(
            readability_score=readability,
            maintainability_score=maintainability,
            efficiency_score=efficiency,
            security_score=security,
            testability_score=testability,
            overall_score=overall
        )
        
        is_valid = len(issues) == 0
        confidence = 0.9 if is_valid else 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            quality_metrics=quality_metrics,
            confidence=confidence
        )
    
    def _validate_javascript_code(self, code: str, requirements: List[str]) -> ValidationResult:
        """Validate JavaScript/TypeScript code and assess quality."""
        issues = []
        suggestions = []
        
        # Basic syntax checks
        if code.count("(") != code.count(")"):
            issues.append("Mismatched parentheses")
        if code.count("{") != code.count("}"):
            issues.append("Mismatched braces")
        if code.count("[") != code.count("]"):
            issues.append("Mismatched brackets")
        
        # Check for required elements
        missing_elements = []
        for req in requirements:
            if req.lower() not in code.lower():
                missing_elements.append(req)
        
        if missing_elements:
            issues.append(f"Missing required elements: {', '.join(missing_elements)}")
        
        # Quality metrics calculation
        readability = self._assess_readability(code)
        maintainability = self._assess_maintainability(code)
        efficiency = self._assess_efficiency(code)
        security = self._assess_security(code)
        testability = self._assess_testability(code)
        
        overall = (readability + maintainability + efficiency + security + testability) / 5
        
        # Add suggestions based on quality scores
        if readability < 0.7:
            suggestions.append("Improve code readability by using descriptive variable names and adding comments")
        if maintainability < 0.7:
            suggestions.append("Improve maintainability by breaking down large functions and reducing complexity")
        if efficiency < 0.7:
            suggestions.append("Optimize code efficiency by reviewing algorithms and data structures")
        if security < 0.7:
            suggestions.append("Enhance security by validating inputs and handling exceptions properly")
        if testability < 0.7:
            suggestions.append("Improve testability by writing modular code and adding unit tests")
        
        # Check for JavaScript-specific best practices
        if "const " not in code and "let " not in code:
            suggestions.append("Use const or let instead of var for variable declarations")
        
        if "=>" not in code and "function" not in code:
            suggestions.append("Consider using arrow functions for concise syntax")
        
        if "console.log" in code:
            suggestions.append("Consider using proper logging instead of console.log for production code")
        
        # Check for proper error handling
        if "try" not in code and "catch" not in code:
            suggestions.append("Add proper error handling with try/catch blocks")
        
        quality_metrics = QualityMetrics(
            readability_score=readability,
            maintainability_score=maintainability,
            efficiency_score=efficiency,
            security_score=security,
            testability_score=testability,
            overall_score=overall
        )
        
        is_valid = len(issues) == 0
        confidence = 0.9 if is_valid else 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            quality_metrics=quality_metrics,
            confidence=confidence
        )
    
    def _validate_go_code(self, code: str, requirements: List[str]) -> ValidationResult:
        """Validate Go code and assess quality."""
        issues = []
        suggestions = []
        
        # Basic syntax checks
        if code.count("(") != code.count(")"):
            issues.append("Mismatched parentheses")
        if code.count("{") != code.count("}"):
            issues.append("Mismatched braces")
        if code.count("[") != code.count("]"):
            issues.append("Mismatched brackets")
        
        # Check for required elements
        missing_elements = []
        for req in requirements:
            if req.lower() not in code.lower():
                missing_elements.append(req)
        
        if missing_elements:
            issues.append(f"Missing required elements: {', '.join(missing_elements)}")
        
        # Quality metrics calculation
        readability = self._assess_readability(code)
        maintainability = self._assess_maintainability(code)
        efficiency = self._assess_efficiency(code)
        security = self._assess_security(code)
        testability = self._assess_testability(code)
        
        overall = (readability + maintainability + efficiency + security + testability) / 5
        
        # Add suggestions based on quality scores
        if readability < 0.7:
            suggestions.append("Improve code readability by using descriptive variable names and adding comments")
        if maintainability < 0.7:
            suggestions.append("Improve maintainability by breaking down large functions and reducing complexity")
        if efficiency < 0.7:
            suggestions.append("Optimize code efficiency by reviewing algorithms and data structures")
        if security < 0.7:
            suggestions.append("Enhance security by validating inputs and handling exceptions properly")
        if testability < 0.7:
            suggestions.append("Improve testability by writing modular code and adding unit tests")
        
        # Check for Go-specific best practices
        if "package " not in code:
            suggestions.append("Add package declaration at the beginning of the file")
        
        if "import " not in code:
            suggestions.append("Add import statements for required packages")
        
        if "func " in code and "return" not in code:
            suggestions.append("Functions should generally return values or have clear side effects")
        
        # Check for proper error handling
        if "error" in code and "if err !=" not in code:
            suggestions.append("Handle errors properly using if err != nil pattern")
        
        quality_metrics = QualityMetrics(
            readability_score=readability,
            maintainability_score=maintainability,
            efficiency_score=efficiency,
            security_score=security,
            testability_score=testability,
            overall_score=overall
        )
        
        is_valid = len(issues) == 0
        confidence = 0.9 if is_valid else 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            quality_metrics=quality_metrics,
            confidence=confidence
        )
    
    def _validate_generic_code(self, code: str, requirements: List[str]) -> ValidationResult:
        """Validate generic code and assess quality."""
        issues = []
        suggestions = []
        
        # Basic syntax checks
        if code.count("(") != code.count(")"):
            issues.append("Mismatched parentheses")
        if code.count("{") != code.count("}"):
            issues.append("Mismatched braces")
        if code.count("[") != code.count("]"):
            issues.append("Mismatched brackets")
        
        # Check for required elements
        missing_elements = []
        for req in requirements:
            if req.lower() not in code.lower():
                missing_elements.append(req)
        
        if missing_elements:
            issues.append(f"Missing required elements: {', '.join(missing_elements)}")
        
        # Quality metrics calculation
        readability = self._assess_readability(code)
        maintainability = self._assess_maintainability(code)
        efficiency = self._assess_efficiency(code)
        security = self._assess_security(code)
        testability = self._assess_testability(code)
        
        overall = (readability + maintainability + efficiency + security + testability) / 5
        
        # Add general suggestions
        if readability < 0.7:
            suggestions.append("Improve code readability by using descriptive variable names and adding comments")
        if maintainability < 0.7:
            suggestions.append("Improve maintainability by breaking down large functions and reducing complexity")
        if efficiency < 0.7:
            suggestions.append("Optimize code efficiency by reviewing algorithms and data structures")
        if security < 0.7:
            suggestions.append("Enhance security by validating inputs and handling exceptions properly")
        if testability < 0.7:
            suggestions.append("Improve testability by writing modular code")
        
        quality_metrics = QualityMetrics(
            readability_score=readability,
            maintainability_score=maintainability,
            efficiency_score=efficiency,
            security_score=security,
            testability_score=testability,
            overall_score=overall
        )
        
        is_valid = len(issues) == 0
        confidence = 0.8 if is_valid else 0.6
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            quality_metrics=quality_metrics,
            confidence=confidence
        )
    
    def _assess_readability(self, code: str) -> float:
        """Assess code readability."""
        # Simple heuristic: check for comments, line length, and naming
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//'))
        total_lines = len([line for line in lines if line.strip()])
        
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        
        # Check for long lines (generally bad for readability)
        long_lines = sum(1 for line in lines if len(line) > 100)
        long_line_ratio = long_lines / total_lines if total_lines > 0 else 0
        
        # Simple scoring
        score = 0.5 + (comment_ratio * 0.3) - (long_line_ratio * 0.2)
        return max(0.0, min(1.0, score))
    
    def _assess_maintainability(self, code: str) -> float:
        """Assess code maintainability."""
        lines = code.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return 1.0
        
        # Check for very long functions (maintainability issue)
        function_lines = 0
        max_function_lines = 0
        
        for line in lines:
            if line.strip().startswith(('def ', 'function ', 'func ')):
                function_lines = 1
            elif line.strip() and function_lines > 0:
                function_lines += 1
                if line.strip() == '}':
                    max_function_lines = max(max_function_lines, function_lines)
                    function_lines = 0
        
        # Heuristic: functions should ideally be < 50 lines
        if max_function_lines > 50:
            score = 1.0 - (max_function_lines - 50) / 100
        else:
            score = 1.0
            
        return max(0.0, min(1.0, score))
    
    def _assess_efficiency(self, code: str) -> float:
        """Assess code efficiency."""
        # Simple heuristics for efficiency
        efficiency_indicators = [
            ('for.*in range', -0.1),  # Python range loops
            ('for.*\.\.\.', -0.1),   # JavaScript for loops
            ('for.*\{\}', -0.1),     # C-style for loops
            ('while.*true', -0.2),   # Potential infinite loops
            ('.*recursive.*', -0.1), # Recursive functions (may be inefficient)
        ]
        
        score = 1.0
        for pattern, penalty in efficiency_indicators:
            if re.search(pattern, code):
                score += penalty
                
        return max(0.0, min(1.0, score))
    
    def _assess_security(self, code: str) -> float:
        """Assess code security."""
        # Check for common security issues
        security_issues = [
            ('eval\(', -0.3),        # Dangerous eval usage
            ('exec\(', -0.3),        # Dangerous exec usage
            ('subprocess\.', -0.2),  # Potential command injection
            ('os\.system\(', -0.2),  # Potential command injection
        ]
        
        score = 1.0
        for pattern, penalty in security_issues:
            if re.search(pattern, code):
                score += penalty
                
        return max(0.0, min(1.0, score))
    
    def _assess_testability(self, code: str) -> float:
        """Assess code testability."""
        # Check for testing-related features
        test_indicators = [
            ('test', 0.2),
            ('assert', 0.1),
            ('mock', 0.1),
            ('fixture', 0.1),
        ]
        
        score = 0.5  # Base score
        for pattern, bonus in test_indicators:
            if re.search(pattern, code, re.IGNORECASE):
                score += bonus
                
        return max(0.0, min(1.0, score))


class CodeFormatter:
    """Formats and beautifies code."""
    
    def __init__(self):
        self.formatters = {
            "python": self._format_python,
            "javascript": self._format_javascript,
            "typescript": self._format_javascript,
            "go": self._format_go
        }
    
    def format_code(self, code: str, language: str = "python") -> str:
        """Format code for better readability."""
        if language.lower() in self.formatters:
            formatter = self.formatters[language.lower()]
            return formatter(code)
        else:
            return self._format_generic(code)
    
    def _format_python(self, code: str) -> str:
        """Format Python code."""
        # Simple formatting - in practice, we'd use black or autopep8
        lines = code.split('\n')
        formatted_lines = []
        
        indent_level = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
                
            # Decrease indent for closing statements
            if stripped.startswith(('elif', 'else', 'except', 'finally')):
                indent_level = max(0, indent_level - 1)
            
            # Add current line with proper indentation
            if stripped:
                formatted_lines.append('    ' * indent_level + stripped)
            else:
                formatted_lines.append('')
            
            # Increase indent for opening statements
            if stripped.endswith(':') and not stripped.startswith(('elif', 'else')):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_javascript(self, code: str) -> str:
        """Format JavaScript/TypeScript code."""
        # Simple formatting - in practice, we'd use prettier
        lines = code.split('\n')
        formatted_lines = []
        
        indent_level = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Decrease indent for closing braces
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            # Add current line with proper indentation
            if stripped:
                formatted_lines.append('  ' * indent_level + stripped)
            else:
                formatted_lines.append('')
            
            # Increase indent for opening braces
            if '{' in stripped and stripped.endswith('{'):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_go(self, code: str) -> str:
        """Format Go code."""
        # Simple formatting - in practice, we'd use gofmt
        lines = code.split('\n')
        formatted_lines = []
        
        indent_level = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Decrease indent for closing braces
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            # Add current line with proper indentation
            if stripped:
                formatted_lines.append('\t' * indent_level + stripped)
            else:
                formatted_lines.append('')
            
            # Increase indent for opening braces
            if stripped.endswith('{'):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_generic(self, code: str) -> str:
        """Format generic code."""
        # Simple generic formatting
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            formatted_lines.append(line.rstrip())
            
        return '\n'.join(formatted_lines)


# Global instances
output_validator = OutputValidator()
code_formatter = CodeFormatter()