"""
Architecture analyzer - detects repo-specific patterns and generates constraints.
This is the critical stage that prevents generic diagram generation.
"""

from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import re


class ArchitectureAnalyzer:
    """Analyze code structure to detect architectural patterns and constraints."""
    
    def __init__(self, rich_facts: Dict[str, Any]):
        self.rich_facts = rich_facts
        self.pattern = None
        self.components = []
        self.workflows = []
        self.technologies = {}
        self.critical_integrations = []
        self.constraints = {}
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze architecture and generate repo-specific constraints.
        
        Returns:
            {
                "pattern": str,  # detected pattern
                "components": [component_dict],
                "workflows": [workflow_dict],
                "technologies": {...},
                "critical_integrations": [...],
                "constraints": {...}
            }
        """
        # Step 1: Detect architectural pattern
        self._detect_pattern()
        
        # Step 2: Extract components
        self._extract_components()
        
        # Step 3: Trace workflows
        self._trace_workflows()
        
        # Step 4: Detect technologies
        self._detect_technologies()
        
        # Step 5: Extract critical integrations
        self._extract_integrations()
        
        # Step 6: Generate constraints
        self._generate_constraints()
        
        return {
            "pattern": self.pattern,
            "components": self.components,
            "workflows": self.workflows,
            "technologies": self.technologies,
            "critical_integrations": self.critical_integrations,
            "constraints": self.constraints,
        }
    
    def _detect_pattern(self) -> None:
        """Detect architectural pattern from code structure."""
        import_graph = self.rich_facts.get("import_graph", {})
        frameworks = self.rich_facts.get("frameworks", [])
        files = self.rich_facts.get("files", [])
        
        # Count decorators/annotations to detect pattern
        controller_count = 0
        service_count = 0
        repository_count = 0
        model_count = 0
        handler_count = 0
        event_count = 0
        
        for file_fact in files:
            if "error" in file_fact and file_fact["error"]:
                continue
            
            for cls in file_fact.get("classes", []):
                name_lower = cls.get("name", "").lower()
                if "controller" in name_lower or "endpoint" in name_lower:
                    controller_count += 1
                elif "service" in name_lower:
                    service_count += 1
                elif "repository" in name_lower or "dao" in name_lower:
                    repository_count += 1
                elif "model" in name_lower or "entity" in name_lower:
                    model_count += 1
                elif "handler" in name_lower:
                    handler_count += 1
                elif "event" in name_lower:
                    event_count += 1
        
        # Heuristics to determine pattern
        if "nestjs" in frameworks or "spring" in frameworks or (controller_count > 0 and service_count > 0 and repository_count > 0):
            self.pattern = "mvc"
        elif event_count > controller_count:
            self.pattern = "event_driven"
        elif len(import_graph) > 10 and self._has_circular_deps(import_graph):
            self.pattern = "microservices"
        elif handler_count > controller_count:
            self.pattern = "pipeline"
        elif self._is_layered(import_graph):
            self.pattern = "layered"
        else:
            self.pattern = "monolithic"
    
    def _extract_components(self) -> None:
        """Extract components based on detected pattern and module structure."""
        files = self.rich_facts.get("files", [])
        import_graph = self.rich_facts.get("import_graph", {})
        
        # Group files by responsibility
        component_map = defaultdict(set)
        
        for file_fact in files:
            if "error" in file_fact and file_fact["error"]:
                continue
            
            file_path = file_fact.get("file", "")
            
            # Classify by directory/name patterns
            if "agent" in file_path.lower():
                component = "Agent Layer"
                component_type = "agent"
            elif "controller" in file_path.lower():
                component = "API Layer"
                component_type = "entry_point"
            elif "router" in file_path.lower():
                component = "API Layer"
                component_type = "entry_point"
            elif "route" in file_path.lower():
                component = "API Layer"
                component_type = "entry_point"
            elif "api" in file_path.lower():
                component = "API Layer"
                component_type = "entry_point"
            elif "service" in file_path.lower():
                component = "Service Layer"
                component_type = "processor"
            elif "extract" in file_path.lower():
                component = "Extraction Layer"
                component_type = "processor"
            elif "rag" in file_path.lower() or "index" in file_path.lower():
                component = "RAG Layer"
                component_type = "datastore"
            elif "model" in file_path.lower():
                component = "Model Layer"
                component_type = "datastore"
            elif "output" in file_path.lower() or "render" in file_path.lower():
                component = "Output Layer"
                component_type = "io"
            elif "core" in file_path.lower():
                component = "Core Layer"
                component_type = "processor"
            else:
                component = "Application Logic"
                component_type = "processor"
            
            component_map[component].add(file_path)
        
        # Build components list
        for component_name, modules in component_map.items():
            responsibility = self._infer_responsibility(component_name, modules)
            component_type = self._infer_type(component_name)
            
            self.components.append({
                "name": component_name,
                "responsibility": responsibility,
                "modules": sorted(list(modules)),
                "type": component_type,
            })
    
    def _trace_workflows(self) -> None:
        """Trace critical workflows from entry points."""
        entry_points = self.rich_facts.get("entry_points", [])
        files = self.rich_facts.get("files", [])
        
        # Find routes in entry points or API layer
        routes = []
        for file_fact in files:
            if "error" in file_fact and file_fact["error"]:
                continue
            routes.extend(file_fact.get("routes", []))
        
        # Create workflow for each major route
        for idx, route in enumerate(routes[:3]):  # Limit to top 3 workflows
            workflow = {
                "name": f"{route['method'].lower()}_{route['path'].replace('/', '_')}",
                "entry": f"{route['method']} {route['path']}",
                "path": self._trace_call_path(route.get("handler", ""), files),
                "criticality": "high" if idx == 0 else "medium",
            }
            self.workflows.append(workflow)
        
        # If no routes, create generic workflow
        if not self.workflows and entry_points:
            self.workflows.append({
                "name": "main_workflow",
                "entry": entry_points[0],
                "path": [entry_points[0]],
                "criticality": "high",
            })
    
    def _detect_technologies(self) -> None:
        """Detect technology stack."""
        frameworks = self.rich_facts.get("frameworks", [])
        files = self.rich_facts.get("files", [])
        
        # Determine framework
        framework = frameworks[0] if frameworks else "unknown"
        
        # Determine async model
        async_count = 0
        total_funcs = 0
        for file_fact in files:
            if "error" in file_fact and file_fact["error"]:
                continue
            for func in file_fact.get("functions", []):
                total_funcs += 1
                if func.get("is_async"):
                    async_count += 1
        
        async_model = "async/await" if async_count > total_funcs / 2 else "callback-based" if async_count == 0 else "mixed"
        
        self.technologies = {
            "framework": framework,
            "orm": "sqlalchemy" if "sqlalchemy" in frameworks else "entity_framework" if "entity_framework" in frameworks else "N/A",
            "async_model": async_model,
            "message_queue": "rabbitmq" if any("rabbitmq" in f.lower() for f in frameworks) else "kafka" if any("kafka" in f.lower() for f in frameworks) else None,
            "language": self.rich_facts.get("language", "unknown"),
        }
    
    def _extract_integrations(self) -> None:
        """Extract critical external integrations."""
        imports = set()
        files = self.rich_facts.get("files", [])
        
        for file_fact in files:
            if "error" in file_fact and file_fact["error"]:
                continue
            imports.update(file_fact.get("imports", []))
        
        # Common external integrations
        external_map = {
            "requests": {"name": "HTTP Client", "type": "external_api"},
            "httpx": {"name": "Async HTTP", "type": "external_api"},
            "urllib": {"name": "URL Opener", "type": "external_api"},
            "boto3": {"name": "AWS SDK", "type": "external_api"},
            "redis": {"name": "Redis Cache", "type": "datastore"},
            "postgres": {"name": "PostgreSQL", "type": "datastore"},
            "mysql": {"name": "MySQL", "type": "datastore"},
            "mongodb": {"name": "MongoDB", "type": "datastore"},
            "faiss": {"name": "FAISS Vector DB", "type": "library"},
            "gemini": {"name": "Gemini LLM", "type": "external_api"},
            "openai": {"name": "OpenAI API", "type": "external_api"},
            "anthropic": {"name": "Claude API", "type": "external_api"},
        }
        
        for import_name in imports:
            for keyword, integration_info in external_map.items():
                if keyword in import_name.lower():
                    if integration_info not in self.critical_integrations:
                        self.critical_integrations.append(integration_info)
    
    def _generate_constraints(self) -> None:
        """Generate constraints for agents to follow."""
        num_files = len([f for f in self.rich_facts.get("files", []) if "error" not in f or not f["error"]])
        num_components = len(self.components)
        
        # Derive constraints from complexity
        max_hld_containers = min(max(num_components, 5), 8)
        max_lld_classes = min(num_files * 2, 15)
        
        # Determine must-include layers based on pattern
        must_include_layers = []
        if self.pattern == "mvc":
            must_include_layers = ["entry_point", "processor", "datastore"]
        elif self.pattern == "layered":
            must_include_layers = ["entry_point", "processor", "datastore"]
        elif self.pattern == "microservices":
            must_include_layers = ["entry_point", "external_api", "processor"]
        elif self.pattern == "pipeline":
            must_include_layers = ["entry_point", "processor", "io"]
        elif self.pattern == "event_driven":
            must_include_layers = ["entry_point", "processor", "message_queue"]
        
        # Determine must-highlight workflows
        must_highlight = [w["name"] for w in self.workflows[:2]]
        
        self.constraints = {
            "max_hld_containers": max_hld_containers,
            "max_lld_class_diagram_classes": max_lld_classes,
            "max_lld_sequence_actors": min(8, num_components),
            "must_include_layers": must_include_layers,
            "must_highlight_workflows": must_highlight,
            "pattern": self.pattern,
        }
    
    # Helper methods
    
    def _is_layered(self, import_graph: Dict) -> bool:
        """Check if import graph suggests layered architecture."""
        # Simple heuristic: no cycles, clear directional flow
        return len(import_graph) > 2 and not self._has_circular_deps(import_graph)
    
    def _has_circular_deps(self, import_graph: Dict) -> bool:
        """Check for circular dependencies."""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in import_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in import_graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False
    
    def _trace_call_path(self, handler: str, files: List) -> List[str]:
        """Trace call path for a handler."""
        # Simplified: just return entry point
        return [handler] if handler else []
    
    def _infer_responsibility(self, component_name: str, modules: Set) -> str:
        """Infer component responsibility from name and modules."""
        responsibilities = {
            "API Layer": "Handle HTTP requests and route them",
            "Service Layer": "Implement business logic",
            "Model Layer": "Define data models and schemas",
            "Extraction Layer": "Parse and extract code structure",
            "RAG Layer": "Retrieve and index information",
            "Agent Layer": "Run LLM agents",
            "Output Layer": "Generate and render output",
            "Core Layer": "Provide core utilities",
        }
        return responsibilities.get(component_name, "Handle application logic")
    
    def _infer_type(self, component_name: str) -> str:
        """Infer component type from name."""
        type_map = {
            "API Layer": "entry_point",
            "Service Layer": "processor",
            "Model Layer": "datastore",
            "Extraction Layer": "processor",
            "RAG Layer": "datastore",
            "Agent Layer": "agent",
            "Output Layer": "io",
            "Core Layer": "processor",
        }
        return type_map.get(component_name, "processor")


def analyze_architecture(rich_facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to analyze architecture and generate constraints.
    
    Args:
        rich_facts: RichFacts dictionary from extraction
        
    Returns:
        ArchitectureContext with patterns and constraints
    """
    analyzer = ArchitectureAnalyzer(rich_facts)
    return analyzer.analyze()
