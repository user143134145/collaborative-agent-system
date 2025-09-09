"""Automated testing framework for autonomous coding agent."""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class TestGenerator:
    """Generates automated tests for different programming languages."""
    
    def __init__(self):
        self.test_frameworks = {
            "python": ["pytest", "unittest"],
            "javascript": ["jest", "mocha"],
            "typescript": ["jest", "mocha"],
            "go": ["testing"],
            "java": ["junit"],
            "rust": ["cargo-test"]
        }
    
    def generate_tests(self, code: str, language: str = "python", framework: str = "auto") -> Dict[str, str]:
        """Generate test files for the given code."""
        if framework == "auto":
            framework = self._select_framework(language)
            
        if language.lower() == "python":
            return self._generate_python_tests(code, framework)
        elif language.lower() in ["javascript", "typescript"]:
            return self._generate_js_tests(code, framework, language)
        elif language.lower() == "go":
            return self._generate_go_tests(code)
        else:
            return self._generate_basic_tests(code, language)
    
    def _select_framework(self, language: str) -> str:
        """Select the most appropriate testing framework for the language."""
        frameworks = self.test_frameworks.get(language.lower(), ["unittest"])
        # Prefer the first framework in the list
        return frameworks[0]
    
    def _generate_python_tests(self, code: str, framework: str = "pytest") -> Dict[str, str]:
        """Generate Python tests."""
        test_files = {}
        
        # Extract function names and classes
        functions = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$", code)
        classes = re.findall(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
        
        if framework == "pytest":
            test_content = self._generate_pytest_content(code, functions, classes)
        else:
            test_content = self._generate_unittest_content(code, functions, classes)
            
        test_files["test_main.py"] = test_content
        return test_files
    
    def _generate_pytest_content(self, code: str, functions: List[str], classes: List[str]) -> str:
        """Generate pytest-style test content."""
        lines = [
            "# Generated pytest tests",
            "import pytest",
            ""
        ]
        
        # Import the module to be tested
        lines.append("from main import *  # Adjust import as needed")
        lines.append("")
        
        # Generate tests for functions
        for func in functions:
            lines.append(f"def test_{func}():")
            lines.append("    # TODO: Add test implementation")
            lines.append("    assert True  # Placeholder")
            lines.append("")
        
        # Generate tests for classes
        for cls in classes:
            lines.append(f"class Test{cls}:")
            lines.append("    def test_initialization(self):")
            lines.append("        # TODO: Add test implementation")
            lines.append("        assert True  # Placeholder")
            lines.append("")
        
        # Add a basic test
        lines.append("def test_example():")
        lines.append("    # Example test - replace with actual tests")
        lines.append("    assert 1 + 1 == 2")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_unittest_content(self, code: str, functions: List[str], classes: List[str]) -> str:
        """Generate unittest-style test content."""
        lines = [
            "# Generated unittest tests",
            "import unittest",
            ""
        ]
        
        # Import the module to be tested
        lines.append("from main import *  # Adjust import as needed")
        lines.append("")
        lines.append("class TestMain(unittest.TestCase):")
        lines.append("")
        
        # Generate tests for functions
        for func in functions:
            lines.append(f"    def test_{func}(self):")
            lines.append("        # TODO: Add test implementation")
            lines.append("        self.assertTrue(True)  # Placeholder")
            lines.append("")
        
        # Generate tests for classes
        for cls in classes:
            lines.append(f"    def test_{cls.lower()}_initialization(self):")
            lines.append("        # TODO: Add test implementation")
            lines.append("        self.assertTrue(True)  # Placeholder")
            lines.append("")
        
        # Add main execution
        lines.append("if __name__ == '__main__':")
        lines.append("    unittest.main()")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_js_tests(self, code: str, framework: str, language: str) -> Dict[str, str]:
        """Generate JavaScript/TypeScript tests."""
        test_files = {}
        
        # Extract function names
        functions = re.findall(r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
        arrow_functions = re.findall(r"const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*$", code)
        functions.extend(arrow_functions)
        
        if framework == "jest":
            test_content = self._generate_jest_content(code, functions, language)
        else:
            test_content = self._generate_mocha_content(code, functions, language)
            
        test_files["test_main.js" if language == "javascript" else "test_main.ts"] = test_content
        return test_files
    
    def _generate_jest_content(self, code: str, functions: List[str], language: str) -> str:
        """Generate Jest-style test content."""
        ext = "js" if language == "javascript" else "ts"
        
        lines = [
            f"// Generated Jest tests for {language}",
            f"const main = require('./main.{ext}');",
            ""
        ]
        
        # Generate tests for functions
        for func in functions:
            lines.append(f"describe('{func}', () => {{")
            lines.append(f"  test('should work correctly', () => {{")
            lines.append("    // TODO: Add test implementation")
            lines.append("    expect(true).toBe(true);  # Placeholder")
            lines.append("  });")
            lines.append("});")
            lines.append("")
        
        # Add a basic test
        lines.append("test('example test', () => {")
        lines.append("  expect(1 + 1).toBe(2);")
        lines.append("});")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_go_tests(self, code: str) -> Dict[str, str]:
        """Generate Go tests."""
        test_files = {}
        
        # Extract function names
        functions = re.findall(r"func\s+([A-Z][a-zA-Z0-9_]*)", code)
        
        lines = [
            "// Generated Go tests",
            "package main",
            "",
            'import "testing"',
            ""
        ]
        
        # Generate tests for functions
        for func in functions:
            lines.append(f"func Test{func}(t *testing.T) {{")
            lines.append("    // TODO: Add test implementation")
            lines.append("    // Placeholder assertion")
            lines.append("    if 1+1 != 2 {")
            lines.append('        t.Errorf("Basic math failed")')
            lines.append("    }")
            lines.append("}")
            lines.append("")
        
        test_files["main_test.go"] = "\n".join(lines)
        return test_files
    
    def _generate_basic_tests(self, code: str, language: str) -> Dict[str, str]:
        """Generate basic test templates for other languages."""
        test_files = {}
        
        lines = [
            f"# Generated basic tests for {language}",
            "# TODO: Add actual test implementations",
            ""
        ]
        
        test_files[f"test_main.{language[:2]}"] = "\n".join(lines)
        return test_files


class TestRunner:
    """Runs tests and analyzes results."""
    
    def __init__(self):
        self.test_commands = {
            "python": {
                "pytest": "python -m pytest -v",
                "unittest": "python -m unittest discover -v"
            },
            "javascript": {
                "jest": "npm test",
                "mocha": "npx mocha test/"
            },
            "typescript": {
                "jest": "npm test",
                "mocha": "npx mocha test/"
            },
            "go": {
                "testing": "go test -v ./..."
            }
        }
    
    def get_test_command(self, language: str, framework: str = "auto") -> str:
        """Get the appropriate test command for the language and framework."""
        if language.lower() not in self.test_commands:
            return "echo 'No test command available'"
            
        frameworks = self.test_commands[language.lower()]
        
        if framework == "auto":
            # Use the first available framework
            framework = next(iter(frameworks))
            
        return frameworks.get(framework, next(iter(frameworks)))
    
    def analyze_test_results(self, output: str, exit_code: int) -> Dict[str, Any]:
        """Analyze test output and return results."""
        results = {
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "errors": [],
            "summary": ""
        }
        
        # Simple parsing - in a real implementation, we'd want more sophisticated parsing
        if "python" in output.lower():
            # Parse pytest/unittest output
            passed_match = re.search(r"(\d+)\s+passed", output)
            failed_match = re.search(r"(\d+)\s+failed", output)
            total_match = re.search(r"(\d+)\s+tests?", output)
            
            if passed_match:
                results["passed_tests"] = int(passed_match.group(1))
            if failed_match:
                results["failed_tests"] = int(failed_match.group(1))
            if total_match:
                results["total_tests"] = int(total_match.group(1))
                
        elif "jest" in output.lower() or "mocha" in output.lower():
            # Parse JavaScript test output
            passed_match = re.search(r"(\d+)\s+passed", output)
            failed_match = re.search(r"(\d+)\s+failed", output)
            
            if passed_match:
                results["passed_tests"] = int(passed_match.group(1))
            if failed_match:
                results["failed_tests"] = int(failed_match.group(1))
                
        # Generate summary
        if results["passed"]:
            results["summary"] = f"All tests passed ({results['passed_tests']} tests)"
        else:
            results["summary"] = f"Tests failed: {results['failed_tests']} failed out of {results['total_tests']} total tests"
            
        return results


# Global instances
test_generator = TestGenerator()
test_runner = TestRunner()