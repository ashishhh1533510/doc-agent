"""
Enhanced Python extractor using AST analysis.
Extracts routes, DB models, class fields, method call graphs, and internal dependencies.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict


class PythonExtractor:
    """Extract detailed code structure from Python files."""
    
    def __init__(self):
        self.imports = set()
        self.classes = []
        self.functions = []
        self.routes = []
        self.db_models = []
        self.internal_calls = defaultdict(set)
        self.current_class = None
    
    def extract_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all information from a single Python file.
        
        Returns:
            {
                "file": str,
                "module_docstring": str | None,
                "imports": [str],
                "functions": [function_dict],
                "classes": [class_dict],
                "routes": [route_dict],
                "db_models": [model_dict],
                "internal_calls": {method: [called_methods]},
                "error": str | None
            }
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            return {
                "file": file_path,
                "error": f"Read error: {str(e)}",
            }
        
        # Reset state
        self.imports = set()
        self.classes = []
        self.functions = []
        self.routes = []
        self.db_models = []
        self.internal_calls = defaultdict(set)
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {
                "file": file_path,
                "error": f"SyntaxError: {str(e)}",
            }
        
        # Extract module docstring
        module_docstring = ast.get_docstring(tree)
        
        # Walk the AST
        self._visit_module(tree)
        
        return {
            "file": file_path,
            "module_docstring": module_docstring,
            "imports": sorted(list(self.imports)),
            "functions": self.functions,
            "classes": self.classes,
            "routes": self.routes,
            "db_models": self.db_models,
            "internal_calls": dict(self.internal_calls),
            "error": None,
        }
    
    def _visit_module(self, tree: ast.Module) -> None:
        """Visit module and extract all top-level definitions."""
        for node in tree.body:
            if isinstance(node, ast.Import):
                self._visit_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._visit_import_from(node)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                self.functions.append(self._extract_function(node, None))
            elif isinstance(node, ast.ClassDef):
                self.classes.append(self._extract_class(node))
    
    def _visit_import(self, node: ast.Import) -> None:
        """Extract regular imports (non-stdlib)."""
        for alias in node.names:
            module = alias.name.split(".")[0]
            if not self._is_stdlib(module):
                self.imports.add(module)
    
    def _visit_import_from(self, node: ast.ImportFrom) -> None:
        """Extract from imports (non-stdlib)."""
        if node.module and not self._is_stdlib(node.module):
            self.imports.add(node.module.split(".")[0])
    
    def _extract_function(self, node: ast.FunctionDef, class_name: str | None) -> Dict[str, Any]:
        """Extract function/method information."""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        decorators = [ast.unparse(d) for d in node.decorator_list]
        signature = self._build_signature(node)
        returns = ast.unparse(node.returns) if node.returns else None
        docstring = ast.get_docstring(node)
        
        # Detect routes
        for decorator in decorators:
            if any(keyword in decorator for keyword in ["@app.", "@router.", "@get", "@post", "@put", "@delete", "@patch"]):
                route_method = self._extract_http_method(decorator)
                route_path = self._extract_route_path(decorator)
                if route_path:
                    self.routes.append({
                        "method": route_method,
                        "path": route_path,
                        "handler": f"{class_name}.{node.name}" if class_name else node.name,
                    })
        
        # Extract calls to other methods
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        called = f"{child.func.value.id}.{child.func.attr}"
                        self.internal_calls[node.name].add(called)
                    elif isinstance(child.func.value, ast.Attribute):
                        # Handle nested attribute access
                        base = self._unparse_attribute(child.func.value)
                        called = f"{base}.{child.func.attr}"
                        self.internal_calls[node.name].add(called)
                elif isinstance(child.func, ast.Name):
                    self.internal_calls[node.name].add(child.func.id)
        
        return {
            "name": node.name,
            "is_async": is_async,
            "signature": signature,
            "returns": returns,
            "decorators": decorators,
            "docstring": docstring,
            "lineno": node.lineno,
        }
    
    def _extract_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information including fields and methods."""
        bases = [ast.unparse(base) for base in node.bases]
        docstring = ast.get_docstring(node)
        
        # Detect DB models (inherit from common base classes)
        is_db_model = any(
            keyword in str(base).lower()
            for base in bases
            for keyword in ["model", "base", "sqlalchemy", "orm"]
        )
        
        if is_db_model:
            fields = self._extract_class_fields(node)
            self.db_models.append({
                "name": node.name,
                "bases": bases,
                "fields": fields,
            })
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item, node.name))
        
        return {
            "name": node.name,
            "bases": bases,
            "docstring": docstring,
            "methods": methods,
            "lineno": node.lineno,
        }
    
    def _extract_class_fields(self, node: ast.ClassDef) -> List[str]:
        """Extract class field type annotations."""
        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                annotation = ast.unparse(item.annotation)
                fields.append(f"{item.target.id}: {annotation}")
        return fields
    
    def _build_signature(self, node: ast.FunctionDef) -> str:
        """Build function signature from AST node."""
        args = node.args
        sig_parts = []
        
        # Regular arguments
        for arg in args.args:
            annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
            sig_parts.append(f"{arg.arg}: {annotation}")
        
        # *args
        if args.vararg:
            annotation = ast.unparse(args.vararg.annotation) if args.vararg.annotation else "Any"
            sig_parts.append(f"*{args.vararg.arg}: {annotation}")
        
        # **kwargs
        if args.kwarg:
            annotation = ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else "Any"
            sig_parts.append(f"**{args.kwarg.arg}: {annotation}")
        
        return f"{node.name}({', '.join(sig_parts)})"
    
    def _extract_http_method(self, decorator: str) -> str:
        """Extract HTTP method from decorator string."""
        decorator_lower = decorator.lower()
        for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
            if method in decorator_lower:
                return method.upper()
        return "GET"
    
    def _extract_route_path(self, decorator: str) -> str | None:
        """Extract route path from decorator string."""
        import re
        # Try to find quoted strings
        match = re.search(r'["\']([^"\']*)["\']', decorator)
        if match:
            return match.group(1)
        return None
    
    def _unparse_attribute(self, node: ast.Attribute) -> str:
        """Unparse nested attribute nodes."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._unparse_attribute(node.value)}.{node.attr}"
        return ast.unparse(node)
    
    @staticmethod
    def _is_stdlib(module_name: str) -> bool:
        """Check if a module is from Python standard library."""
        stdlib_modules = {
            "os", "sys", "re", "json", "ast", "typing", "collections", "functools",
            "itertools", "math", "random", "datetime", "time", "pathlib", "logging",
            "threading", "asyncio", "subprocess", "pickle", "enum", "dataclasses",
            "abc", "contextlib", "io", "urllib", "http", "ssl", "socket", "email",
            "csv", "xml", "html", "base64", "hashlib", "hmac", "secrets", "string",
            "textwrap", "unicodedata", "struct", "codecs", "tempfile", "glob", "fnmatch",
            "linecache", "shutil", "gzip", "zipfile", "tarfile", "dbm", "sqlite3",
            "unittest", "doctest", "pdb", "profile", "pstats", "timeit", "trace",
            "inspect", "types", "weakref", "gc", "copy", "copyreg", "types",
        }
        return module_name in stdlib_modules


def extract_from_python_file(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to extract information from a single Python file.
    """
    extractor = PythonExtractor()
    return extractor.extract_file(file_path)


def extract_from_python_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Extract information from all Python files in a directory.
    """
    extractor = PythonExtractor()
    results = []
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__", ".git", "node_modules"}]
        
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                result = extractor.extract_file(file_path)
                results.append(result)
    
    return results
