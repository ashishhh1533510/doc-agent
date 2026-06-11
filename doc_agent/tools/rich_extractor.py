"""
Rich extractor - orchestrates language-specific extraction and builds RichFacts with import graph.
Produces structured, bounded facts grounded in actual code.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict

from .language_detector import detect_language
from .extractors import (
    extract_from_python_file,
    extract_from_typescript_file,
    extract_from_java_file,
    extract_from_csharp_file,
)


class RichExtractor:
    """Extract comprehensive code facts with language awareness."""
    
    EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".idea", "dist", "build", ".next", "target", "bin", "obj"}
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.language_info = {}
        self.files = []
        self.import_graph = defaultdict(set)
        self.frameworks = set()
        self.entry_points = []
    
    def extract(self) -> Dict[str, Any]:
        """
        Extract all code facts from a project.
        
        Returns:
            {
                "project_root": str,
                "language": str,
                "files": [file_facts],
                "import_graph": {module: [dependencies]},
                "frameworks": [str],
                "entry_points": [str]
            }
        """
        # Detect language
        self.language_info = detect_language(str(self.project_root))
        dominant_language = self.language_info.get("dominant")
        
        # Extract files based on language
        self._extract_files(dominant_language)
        
        # Build import graph
        self._build_import_graph()
        
        # Detect frameworks
        self._detect_frameworks()
        
        # Detect entry points
        self._detect_entry_points()
        
        return {
            "project_root": str(self.project_root),
            "language": dominant_language,
            "files": self.files,
            "import_graph": dict(self.import_graph),
            "frameworks": sorted(list(self.frameworks)),
            "entry_points": self.entry_points,
        }
    
    def _extract_files(self, language: str) -> None:
        """Extract files based on detected language."""
        extractor_func = {
            "python": extract_from_python_file,
            "typescript": extract_from_typescript_file,
            "javascript": extract_from_typescript_file,
            "java": extract_from_java_file,
            "csharp": extract_from_csharp_file,
        }.get(language, extract_from_python_file)
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS]
            
            for file in files:
                if language == "python" and file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    result = extractor_func(file_path)
                    self.files.append(result)
                elif language in ["typescript", "javascript"] and file.endswith((".ts", ".tsx", ".js", ".jsx")):
                    file_path = os.path.join(root, file)
                    result = extractor_func(file_path)
                    self.files.append(result)
                elif language == "java" and file.endswith(".java"):
                    file_path = os.path.join(root, file)
                    result = extractor_func(file_path)
                    self.files.append(result)
                elif language == "csharp" and file.endswith(".cs"):
                    file_path = os.path.join(root, file)
                    result = extractor_func(file_path)
                    self.files.append(result)
    
    def _build_import_graph(self) -> None:
        """Build inter-module dependency graph."""
        # Create module to file mapping
        module_to_file = {}
        for file_fact in self.files:
            if "error" in file_fact and file_fact["error"]:
                continue
            
            file_path = Path(file_fact["file"])
            # Convert file path to module path
            module_path = str(file_path.parent.relative_to(self.project_root)).replace("\\", ".").replace("/", ".")
            if module_path == ".":
                module_path = file_path.stem
            else:
                module_path = f"{module_path}.{file_path.stem}"
            
            module_to_file[module_path] = file_fact
        
        # Build import graph
        for file_fact in self.files:
            if "error" in file_fact and file_fact["error"]:
                continue
            
            file_path = Path(file_fact["file"])
            module_path = str(file_path.parent.relative_to(self.project_root)).replace("\\", ".").replace("/", ".")
            if module_path == ".":
                module_path = file_path.stem
            else:
                module_path = f"{module_path}.{file_path.stem}"
            
            for imported in file_fact.get("imports", []):
                self.import_graph[module_path].add(imported)
    
    def _detect_frameworks(self) -> None:
        """Detect frameworks and technologies used."""
        framework_keywords = {
            "fastapi": ["fastapi", "starlette"],
            "flask": ["flask"],
            "django": ["django"],
            "sqlalchemy": ["sqlalchemy", "orm"],
            "nestjs": ["@nestjs", "@Controller"],
            "express": ["express"],
            "spring": ["@SpringBootApplication", "@Controller"],
            "aspnet": ["ASP.NET", "dotnet"],
            "entity_framework": ["DbContext"],
        }
        
        # Search for framework patterns in imports and decorators
        for file_fact in self.files:
            if "error" in file_fact and file_fact["error"]:
                continue
            
            for import_name in file_fact.get("imports", []):
                for framework, keywords in framework_keywords.items():
                    if any(keyword.lower() in import_name.lower() for keyword in keywords):
                        self.frameworks.add(framework)
            
            for func in file_fact.get("functions", []):
                for decorator in func.get("decorators", []):
                    for framework, keywords in framework_keywords.items():
                        if any(keyword.lower() in decorator.lower() for keyword in keywords):
                            self.frameworks.add(framework)
            
            for cls in file_fact.get("classes", []):
                for base in cls.get("bases", []):
                    for framework, keywords in framework_keywords.items():
                        if any(keyword.lower() in base.lower() for keyword in keywords):
                            self.frameworks.add(framework)
    
    def _detect_entry_points(self) -> None:
        """Detect entry points (main files, app.py, index.ts, etc.)."""
        entry_point_names = {
            "main.py", "app.py", "__main__.py",
            "main.ts", "index.ts", "server.ts",
            "Main.java",
            "Program.cs",
            "index.js", "main.js"
        }
        
        for file_fact in self.files:
            if "error" in file_fact and file_fact["error"]:
                continue
            
            file_name = Path(file_fact["file"]).name
            if file_name in entry_point_names:
                self.entry_points.append(file_fact["file"])


def extract_rich_facts(project_root: str) -> Dict[str, Any]:
    """
    Convenience function to extract rich facts from a project.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        RichFacts dictionary with all code structure information
    """
    extractor = RichExtractor(project_root)
    return extractor.extract()
