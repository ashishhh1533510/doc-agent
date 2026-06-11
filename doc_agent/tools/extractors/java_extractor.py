"""
Java extractor using regex patterns.
Best-effort parsing for annotations, classes, and imports.
"""

import re
from typing import Dict, List, Any


class JavaExtractor:
    """Extract code structure from Java files."""
    
    def __init__(self):
        self.imports = set()
        self.classes = []
        self.functions = []
        self.routes = []
    
    def extract_file(self, file_path: str) -> Dict[str, Any]:
        """Extract code structure from Java file."""
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
        
        # Extract imports
        self._extract_imports(source)
        
        # Extract classes
        self._extract_classes(source)
        
        # Extract methods
        self._extract_methods(source)
        
        # Extract routes (Spring annotations)
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
        """Extract import statements."""
        # import com.example.ClassName;
        pattern = r"import\s+([a-zA-Z0-9_.]+)(?:\.\*)?;"
        for match in re.finditer(pattern, source):
            import_path = match.group(1)
            # Get first part of package
            package = import_path.split(".")[0]
            if package not in {"java", "javax", "org.w3c", "sun"}:
                self.imports.add(package)
    
    def _extract_classes(self, source: str) -> None:
        """Extract class definitions."""
        # public class ClassName extends BaseClass
        pattern = r"(?:public\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?"
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
        # public void methodName() or private String methodName()
        pattern = r"(?:public|private|protected)?\s+\w+\s+(\w+)\s*\("
        for match in re.finditer(pattern, source):
            method_name = match.group(1)
            if method_name not in {"if", "for", "while", "switch", "catch"}:
                self.functions.append({
                    "name": method_name,
                    "is_async": False,
                    "signature": f"{method_name}(...)",
                    "returns": None,
                    "decorators": [],
                    "docstring": None,
                    "lineno": source[:match.start()].count("\n") + 1,
                })
    
    def _extract_routes(self, source: str) -> None:
        """Extract route annotations (Spring)."""
        # @GetMapping("/path"), @PostMapping("/path"), etc.
        pattern = r"@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)\s*\(['\"]([^'\"]*)['\"]"
        for match in re.finditer(pattern, source):
            mapping_type = match.group(1)
            path = match.group(2)
            
            # Convert mapping type to HTTP method
            method_map = {
                "GetMapping": "GET",
                "PostMapping": "POST",
                "PutMapping": "PUT",
                "DeleteMapping": "DELETE",
                "PatchMapping": "PATCH",
                "RequestMapping": "GET",
            }
            method = method_map.get(mapping_type, "GET")
            
            self.routes.append({
                "method": method,
                "path": path,
                "handler": "handler",
            })


def extract_from_java_file(file_path: str) -> Dict[str, Any]:
    """Extract information from a Java file."""
    extractor = JavaExtractor()
    return extractor.extract_file(file_path)
