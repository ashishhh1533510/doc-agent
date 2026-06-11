"""
C# extractor using regex patterns.
Best-effort parsing for attributes, classes, and using statements.
"""

import re
from typing import Dict, List, Any


class CSharpExtractor:
    """Extract code structure from C# files."""
    
    def __init__(self):
        self.imports = set()
        self.classes = []
        self.functions = []
        self.routes = []
    
    def extract_file(self, file_path: str) -> Dict[str, Any]:
        """Extract code structure from C# file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            return {
                "file": file_path,
                "error": f"Read error: {str(e)}",
            }
        
        self.imports = set()
        self.classes = []
        self.functions = []
        self.routes = []
        
        # Extract using statements
        self._extract_imports(source)
        
        # Extract classes
        self._extract_classes(source)
        
        # Extract methods
        self._extract_methods(source)
        
        # Extract routes (ASP.NET attributes)
        self._extract_routes(source)
        
        return {
            "file": file_path,
            "module_docstring": None,
            "imports": sorted(list(self.imports)),
            "functions": self.functions,
            "classes": self.classes,
            "routes": self.routes,
            "db_models": [],
            "internal_calls": {},
            "error": None,
        }
    
    def _extract_imports(self, source: str) -> None:
        """Extract using statements."""
        # using System; using MyNamespace.SubNamespace;
        pattern = r"using\s+([a-zA-Z0-9_.]+);"
        for match in re.finditer(pattern, source):
            namespace = match.group(1)
            # Get first part of namespace
            package = namespace.split(".")[0]
            if package not in {"System", "Microsoft"}:
                self.imports.add(package)
    
    def _extract_classes(self, source: str) -> None:
        """Extract class definitions."""
        # public class ClassName : BaseClass
        pattern = r"(?:public|private|protected)?\s+class\s+(\w+)(?:\s*:\s*(\w+))?"
        for match in re.finditer(pattern, source):
            class_name = match.group(1)
            base = match.group(2)
            bases = [base] if base else []
            self.classes.append({
                "name": class_name,
                "bases": bases,
                "docstring": None,
                "methods": [],
                "lineno": source[:match.start()].count("\n") + 1,
            })
    
    def _extract_methods(self, source: str) -> None:
        """Extract method definitions."""
        # public void MethodName() or public async Task MethodName()
        pattern = r"(?:public|private|protected)?\s+(?:async\s+)?(?:\w+\s+)*(\w+)\s*\("
        for match in re.finditer(pattern, source):
            method_name = match.group(1)
            if method_name not in {"if", "for", "while", "switch", "catch", "using"}:
                is_async = "async" in source[max(0, match.start()-30):match.start()]
                self.functions.append({
                    "name": method_name,
                    "is_async": is_async,
                    "signature": f"{method_name}(...)",
                    "returns": None,
                    "decorators": [],
                    "docstring": None,
                    "lineno": source[:match.start()].count("\n") + 1,
                })
    
    def _extract_routes(self, source: str) -> None:
        """Extract route attributes (ASP.NET Core)."""
        # [HttpGet("/path")], [HttpPost("/path")], etc.
        pattern = r"\[(Http(?:Get|Post|Put|Delete|Patch|Head|Options))\s*\(['\"]([^'\"]*)['\"]"
        for match in re.finditer(pattern, source):
            http_type = match.group(1)
            path = match.group(2)
            
            # Extract method from attribute name
            method = http_type.replace("Http", "").upper()
            
            self.routes.append({
                "method": method,
                "path": path,
                "handler": "handler",
            })


def extract_from_csharp_file(file_path: str) -> Dict[str, Any]:
    """Extract information from a C# file."""
    extractor = CSharpExtractor()
    return extractor.extract_file(file_path)
