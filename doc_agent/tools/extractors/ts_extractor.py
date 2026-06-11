"""
TypeScript/JavaScript extractor using regex patterns.
Best-effort parsing for route handlers, classes, and imports.
"""

import re
from typing import Dict, List, Any


class TypeScriptExtractor:
    """Extract code structure from TypeScript/JavaScript files."""
    
    def __init__(self):
        self.imports = set()
        self.classes = []
        self.functions = []
        self.routes = []
    
    def extract_file(self, file_path: str) -> Dict[str, Any]:
        """Extract code structure from TypeScript/JavaScript file."""
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
        
        # Extract functions
        self._extract_functions(source)
        
        # Extract routes (NestJS, Express decorators)
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
        # import { X } from 'module'
        pattern = r"import\s*{[^}]*}\s*from\s*['\"]([^'\"]+)['\"]"
        for match in re.finditer(pattern, source):
            module = match.group(1).split("/")[0]
            if not module.startswith("."):
                self.imports.add(module)
        
        # import X from 'module'
        pattern = r"import\s+\w+\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(pattern, source):
            module = match.group(1).split("/")[0]
            if not module.startswith("."):
                self.imports.add(module)
    
    def _extract_classes(self, source: str) -> None:
        """Extract class definitions."""
        # export class ClassName
        pattern = r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?"
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
    
    def _extract_functions(self, source: str) -> None:
        """Extract function definitions."""
        # export async function NAME or function NAME
        pattern = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("
        for match in re.finditer(pattern, source):
            func_name = match.group(1)
            is_async = "async" in source[max(0, match.start()-20):match.start()]
            self.functions.append({
                "name": func_name,
                "is_async": is_async,
                "signature": f"{func_name}(...)",
                "returns": None,
                "decorators": [],
                "docstring": None,
                "lineno": source[:match.start()].count("\n") + 1,
            })
    
    def _extract_routes(self, source: str) -> None:
        """Extract route decorators (NestJS, Express)."""
        # @Get('/path'), @Post('/path'), etc.
        pattern = r"@(Get|Post|Put|Delete|Patch|Head|Options)\s*\(['\"]([^'\"]*)['\"]"
        for match in re.finditer(pattern, source):
            method = match.group(1).upper()
            path = match.group(2)
            self.routes.append({
                "method": method,
                "path": path,
                "handler": "handler",  # Can't determine from regex alone
            })
        
        # Express app.get('/path'), app.post('/path'), etc.
        pattern = r"app\.(get|post|put|delete|patch)\s*\(['\"]([^'\"]*)['\"]"
        for match in re.finditer(pattern, source):
            method = match.group(1).upper()
            path = match.group(2)
            self.routes.append({
                "method": method,
                "path": path,
                "handler": "handler",
            })


def extract_from_typescript_file(file_path: str) -> Dict[str, Any]:
    """Extract information from a TypeScript/JavaScript file."""
    extractor = TypeScriptExtractor()
    return extractor.extract_file(file_path)
