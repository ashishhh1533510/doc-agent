"""
Language detection module - identifies primary and secondary languages in a repository.
Counts file extensions and determines dominant language + all supported languages.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict


class LanguageDetector:
    """Detect programming languages in a repository by counting file extensions."""
    
    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".java": "java",
        ".cs": "csharp",
        ".go": "go",
    }
    
    EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".idea", "dist", "build", ".next"}
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.file_counts = defaultdict(int)
        self.detected_languages = {}
    
    def detect(self) -> Dict[str, any]:
        """
        Analyze repository and detect language distribution.
        
        Returns:
            {
                "dominant": "python",  # primary language
                "languages": {
                    "python": 42,
                    "typescript": 5,
                    "javascript": 3
                },
                "total_files": 50,
                "supported_languages": ["python", "typescript", "javascript"]
            }
        """
        self._walk_directory()
        
        if not self.file_counts:
            return {
                "dominant": None,
                "languages": {},
                "total_files": 0,
                "supported_languages": [],
            }
        
        # Find dominant language
        dominant = max(self.file_counts, key=self.file_counts.get)
        
        # Filter to supported languages
        supported = {lang: count for lang, count in self.file_counts.items() if lang}
        
        return {
            "dominant": dominant,
            "languages": dict(supported),
            "total_files": sum(self.file_counts.values()),
            "supported_languages": list(supported.keys()),
        }
    
    def _walk_directory(self) -> None:
        """Walk directory tree and count file extensions."""
        try:
            for root, dirs, files in os.walk(self.project_root):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS]
                
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in self.SUPPORTED_EXTENSIONS:
                        language = self.SUPPORTED_EXTENSIONS[ext]
                        self.file_counts[language] += 1
        except Exception as e:
            print(f"Error walking directory: {e}")
    
    def get_language_for_file(self, file_path: str) -> str:
        """Get language name for a specific file."""
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, None)


def detect_language(project_root: str) -> Dict[str, any]:
    """
    Convenience function to detect language in a repository.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Dictionary with detected language information
    """
    detector = LanguageDetector(project_root)
    return detector.detect()
