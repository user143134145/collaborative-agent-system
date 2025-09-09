"""Project scaffolding system for autonomous coding agent."""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class ProjectScaffolder:
    """Creates project structures and boilerplate code for different types of applications."""
    
    def __init__(self):
        self.project_templates = {
            "python-web-api": self._create_python_web_api_template,
            "python-data-analysis": self._create_python_data_analysis_template,
            "python-cli": self._create_python_cli_template,
            "javascript-web-app": self._create_js_web_app_template,
            "javascript-api": self._create_js_web_app_template,  # Use web app template for JS API as fallback
            "go-web-service": self._create_generic_template,  # Use generic template as fallback
            "rust-cli": self._create_generic_template,  # Use generic template as fallback
            "generic": self._create_generic_template
        }
        
        self.directory_structures = {
            "python": {
                "src/": {
                    "__init__.py": "",
                    "main.py": "# Main application entry point\n"
                },
                "tests/": {
                    "__init__.py": "",
                    "test_main.py": "# Test file\n"
                },
                "docs/": {},
                "config/": {},
                "requirements.txt": "",
                "README.md": "# Project Title\n\n## Description\n\n## Installation\n\n## Usage\n\n## Testing\n",
                ".gitignore": "__pycache__\n*.pyc\n.env\n.pytest_cache/\n.coverage\nhtmlcov/\n",
                ".env.example": "# Environment variables\nDEBUG=True\n"
            },
            "javascript": {
                "src/": {
                    "index.js": "// Main application entry point\n"
                },
                "tests/": {
                    "test.js": "// Test file\n"
                },
                "docs/": {},
                "config/": {},
                "package.json": "{\n  \"name\": \"autonomous-app\",\n  \"version\": \"1.0.0\",\n  \"description\": \"\",\n  \"main\": \"src/index.js\",\n  \"scripts\": {\n    \"start\": \"node src/index.js\",\n    \"test\": \"echo \\\"Error: no test specified\\\" && exit 1\"\n  },\n  \"keywords\": [],\n  \"author\": \"\",\n  \"license\": \"ISC\"\n}",
                "README.md": "# Project Title\n\n## Description\n\n## Installation\n\n## Usage\n\n## Testing\n",
                ".gitignore": "node_modules/\nnpm-debug.log*\n.env\n"
            },
            "go": {
                "cmd/": {
                    "main.go": "// Main application entry point\npackage main\n\nfunc main() {\n\t// TODO: Implement\n}\n"
                },
                "internal/": {},
                "pkg/": {},
                "test/": {},
                "go.mod": "module autonomous-app\n\ngo 1.19\n",
                "README.md": "# Project Title\n\n## Description\n\n## Installation\n\n## Usage\n\n## Testing\n",
                ".gitignore": "bin/\n*.exe\n*.exe~\n*.dll\n*.so\n*.dylib\n"
            }
        }
    
    def create_project_structure(self, project_type: str, project_name: str, 
                               requirements: List[str] = None, 
                               language: str = "python") -> Dict[str, Any]:
        """Create a complete project structure."""
        if project_type in self.project_templates:
            return self.project_templates[project_type](project_name, requirements, language)
        else:
            return self._create_generic_template(project_name, requirements, language)
    
    def _create_python_web_api_template(self, project_name: str, requirements: List[str], language: str) -> Dict[str, Any]:
        """Create a Python web API project template."""
        structure = self.directory_structures["python"].copy()
        
        # Update source files for web API
        structure["src/"]["main.py"] = '''"""
Web API Application
"""

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the API"})

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        
        # Update test file
        structure["tests/"]["test_main.py"] = '''"""
Tests for Web API Application
"""

import pytest
from src.main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test home endpoint"""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"Welcome" in rv.data

def test_health_check(client):
    """Test health check endpoint"""
    rv = client.get('/api/health')
    assert rv.status_code == 200
    assert b"healthy" in rv.data
'''
        
        # Update requirements with web API dependencies
        web_requirements = ["flask", "pytest"]
        if requirements:
            web_requirements.extend(requirements)
        structure["requirements.txt"] = "\n".join(web_requirements)
        
        # Update README
        structure["README.md"] = f'''# {project_name}

## Description
A Python web API application built with Flask.

## Features
- RESTful API endpoints
- Health check endpoint
- JSON response format

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

The API will be available at http://localhost:5000

## Endpoints
- GET / - Welcome message
- GET /api/health - Health check

## Testing
```bash
pytest
```
'''
        
        return structure
    
    def _create_python_data_analysis_template(self, project_name: str, requirements: List[str], language: str) -> Dict[str, Any]:
        """Create a Python data analysis project template."""
        structure = self.directory_structures["python"].copy()
        
        # Update source files for data analysis
        structure["src/"]["main.py"] = '''"""
Data Analysis Application
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)

def analyze_data(df):
    """Perform basic analysis on the data"""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'summary_stats': df.describe().to_dict()
    }
    return summary

def visualize_data(df, column):
    """Create a simple visualization"""
    plt.figure(figsize=(10, 6))
    df[column].hist()
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(f'{column}_distribution.png')
    plt.close()

if __name__ == '__main__':
    print("Data Analysis Application")
    # Add your analysis code here
'''
        
        # Update test file
        structure["tests/"]["test_main.py"] = '''"""
Tests for Data Analysis Application
"""

import pytest
import pandas as pd
import numpy as np
from src.main import load_data, analyze_data

def test_load_data():
    """Test data loading function"""
    # Create sample data
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    # Save to CSV and load
    data.to_csv('test_data.csv', index=False)
    
    loaded_data = load_data('test_data.csv')
    assert isinstance(loaded_data, pd.DataFrame)
    assert loaded_data.shape == (5, 2)
    
    # Clean up
    import os
    os.remove('test_data.csv')

def test_analyze_data():
    """Test data analysis function"""
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    analysis = analyze_data(data)
    assert 'shape' in analysis
    assert 'columns' in analysis
    assert 'dtypes' in analysis
'''
        
        # Update requirements with data analysis dependencies
        data_requirements = ["pandas", "numpy", "matplotlib", "pytest"]
        if requirements:
            data_requirements.extend(requirements)
        structure["requirements.txt"] = "\n".join(data_requirements)
        
        # Update README
        structure["README.md"] = f'''# {project_name}

## Description
A Python data analysis application.

## Features
- Data loading from CSV files
- Basic statistical analysis
- Data visualization capabilities

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

## Dependencies
- pandas: Data manipulation
- numpy: Numerical computing
- matplotlib: Data visualization

## Testing
```bash
pytest
```
'''
        
        return structure
    
    def _create_python_cli_template(self, project_name: str, requirements: List[str], language: str) -> Dict[str, Any]:
        """Create a Python CLI project template."""
        structure = self.directory_structures["python"].copy()
        
        # Update source files for CLI
        structure["src/"]["main.py"] = '''"""
CLI Application
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='CLI Application')
    parser.add_argument('--version', action='version', version='1.0.0')
    parser.add_argument('command', nargs='?', default='help', 
                       help='Command to execute')
    
    args = parser.parse_args()
    
    if args.command == 'help':
        print("Available commands:")
        print("  help    - Show this help message")
        print("  version - Show version")
    elif args.command == 'version':
        print("CLI Application v1.0.0")
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
        
        # Update test file
        structure["tests/"]["test_main.py"] = '''"""
Tests for CLI Application
"""

import pytest
from src.main import main
from unittest.mock import patch

def test_main_help():
    """Test help command"""
    with patch('sys.argv', ['main.py', 'help']):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

def test_main_version():
    """Test version command"""
    with patch('sys.argv', ['main.py', 'version']):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
'''
        
        # Update requirements
        cli_requirements = ["pytest"]
        if requirements:
            cli_requirements.extend(requirements)
        structure["requirements.txt"] = "\n".join(cli_requirements)
        
        # Update README
        structure["README.md"] = f'''# {project_name}

## Description
A Python command-line interface (CLI) application.

## Features
- Command-line argument parsing
- Extensible command structure
- Help and version commands

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py [command]
```

### Commands
- help    - Show help message
- version - Show version information

## Testing
```bash
pytest
```
'''
        
        return structure
    
    def _create_js_web_app_template(self, project_name: str, requirements: List[str], language: str) -> Dict[str, Any]:
        """Create a JavaScript web application template."""
        structure = self.directory_structures["javascript"].copy()
        
        # Update source files for web app
        structure["src/"]["index.js"] = '''/**
 * Web Application
 */

// Express app example
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.get('/api/status', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date() });
});

app.listen(port, () => {
  console.log(`App listening at http://localhost:${port}`);
});
'''
        
        # Update package.json with express
        package_json = {
            "name": project_name.lower().replace(" ", "-"),
            "version": "1.0.0",
            "description": "A JavaScript web application",
            "main": "src/index.js",
            "scripts": {
                "start": "node src/index.js",
                "dev": "nodemon src/index.js",
                "test": "jest"
            },
            "dependencies": {
                "express": "^4.18.0"
            },
            "devDependencies": {
                "nodemon": "^2.0.0",
                "jest": "^29.0.0"
            },
            "keywords": [],
            "author": "",
            "license": "ISC"
        }
        
        if requirements:
            for req in requirements:
                package_json["dependencies"][req] = "^1.0.0"
                
        structure["package.json"] = json.dumps(package_json, indent=2)
        
        # Update README
        structure["README.md"] = f'''# {project_name}

## Description
A JavaScript web application built with Express.

## Features
- Web server with Express
- RESTful API endpoints
- JSON response format

## Installation
```bash
npm install
```

## Usage
```bash
npm start
```

The application will be available at http://localhost:3000

## Endpoints
- GET / - Welcome message
- GET /api/status - Status check

## Development
```bash
npm run dev
```

## Testing
```bash
npm test
```
'''
        
        return structure
    
    def _create_generic_template(self, project_name: str, requirements: List[str], language: str) -> Dict[str, Any]:
        """Create a generic project template."""
        if language.lower() in self.directory_structures:
            structure = self.directory_structures[language.lower()].copy()
        else:
            structure = self.directory_structures["python"].copy()
        
        # Update README
        structure["README.md"] = f'''# {project_name}

## Description
An autonomous application generated by AI.

## Installation
```bash
# Install dependencies based on your language
```

## Usage
```bash
# Run the application
```

## Testing
```bash
# Run tests
```
'''
        
        # Update requirements/package files
        if language.lower() == "python" and requirements:
            structure["requirements.txt"] = "\n".join(requirements)
        elif language.lower() in ["javascript", "typescript"] and requirements:
            package_json = json.loads(structure["package.json"])
            for req in requirements:
                package_json["dependencies"][req] = "^1.0.0"
            structure["package.json"] = json.dumps(package_json, indent=2)
        
        return structure


class FileGenerator:
    """Generates specific files based on templates and requirements."""
    
    def __init__(self):
        self.file_templates = {
            "dockerfile": self._generate_dockerfile,
            "readme": self._generate_readme,
            "gitignore": self._generate_gitignore,
            "config": self._generate_config
        }
    
    def generate_file(self, file_type: str, **kwargs) -> str:
        """Generate a specific type of file."""
        if file_type in self.file_templates:
            return self.file_templates[file_type](**kwargs)
        else:
            return f"# {file_type.capitalize()} file\n"
    
    def _generate_dockerfile(self, language: str = "python", app_name: str = "app") -> str:
        """Generate a Dockerfile."""
        if language.lower() == "python":
            return '''# Dockerfile for Python application
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "src/main.py"]
'''
        elif language.lower() in ["javascript", "typescript"]:
            return '''# Dockerfile for Node.js application
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD ["npm", "start"]
'''
        elif language.lower() == "go":
            return f'''# Dockerfile for Go application
FROM golang:1.19-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o {app_name} cmd/main.go

FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

COPY --from=builder /app/{app_name} .

EXPOSE 8080

CMD ["./{app_name}"]
'''
        else:
            return f"# Dockerfile for {language}\n# TODO: Add implementation"
    
    def _generate_readme(self, project_name: str, description: str = "", 
                        installation: str = "", usage: str = "") -> str:
        """Generate a README file."""
        return f'''# {project_name}

## Description
{description or "An autonomous application generated by AI."}

## Installation
{installation or "```bash\n# Install dependencies\n```"}

## Usage
{usage or "```bash\n# Run the application\n```"}

## Features
- Feature 1
- Feature 2
- Feature 3

## License
MIT
'''
    
    def _generate_gitignore(self, language: str = "python") -> str:
        """Generate a .gitignore file."""
        common_ignores = [
            ".env",
            "*.log",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        if language.lower() == "python":
            python_ignores = [
                "__pycache__/",
                "*.pyc",
                "*.pyo",
                "*.pyd",
                ".Python",
                "*.so",
                ".coverage",
                ".pytest_cache/",
                ".tox/",
                ".eggs/",
                "*.egg-info/",
                "dist/",
                "build/"
            ]
            return "\n".join(common_ignores + python_ignores)
        elif language.lower() in ["javascript", "typescript"]:
            js_ignores = [
                "node_modules/",
                "npm-debug.log*",
                "yarn-debug.log*",
                "yarn-error.log*",
                ".npm/",
                ".yarn-integrity",
                ".yarn/cache"
            ]
            return "\n".join(common_ignores + js_ignores)
        else:
            return "\n".join(common_ignores)
    
    def _generate_config(self, app_name: str, config_type: str = "json") -> str:
        """Generate a configuration file."""
        if config_type.lower() == "json":
            return '''{
  "app_name": "%s",
  "version": "1.0.0",
  "debug": true,
  "port": 5000,
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "app_db"
  }
}''' % app_name
        elif config_type.lower() == "yaml":
            return f'''app_name: {app_name}
version: 1.0.0
debug: true
port: 5000
database:
  host: localhost
  port: 5432
  name: app_db
'''
        else:
            return f"# Configuration for {app_name}\n# TODO: Add implementation"


# Global instances
project_scaffolder = ProjectScaffolder()
file_generator = FileGenerator()