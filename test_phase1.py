#!/usr/bin/env python
"""Test script for Phase 1 extraction and architecture analysis."""

from doc_agent.tools.rich_extractor import extract_rich_facts
from doc_agent.tools.architecture_analyzer import analyze_architecture
import json

# Extract rich facts
print("Extracting rich facts...")
rich_facts = extract_rich_facts('.')

print('\n=== RICH FACTS ===')
print(f'Language: {rich_facts["language"]}')
print(f'Files extracted: {len(rich_facts["files"])}')
print(f'Frameworks: {rich_facts["frameworks"]}')
print(f'Entry points: {rich_facts["entry_points"]}')
print(f'Components in import graph: {len(rich_facts["import_graph"])}')

# Show first few files
print(f'\nFirst 3 files extracted:')
for file_fact in rich_facts["files"][:3]:
    if "error" not in file_fact or not file_fact["error"]:
        print(f'  - {file_fact["file"]}')
        print(f'    Functions: {len(file_fact["functions"])}')
        print(f'    Classes: {len(file_fact["classes"])}')
        print(f'    Routes: {len(file_fact["routes"])}')

# Analyze architecture
print('\nAnalyzing architecture...')
arch_context = analyze_architecture(rich_facts)

print('\n=== ARCHITECTURE CONTEXT ===')
print(f'Pattern detected: {arch_context["pattern"]}')
print(f'\nComponents ({len(arch_context["components"])} total):')
for comp in arch_context['components']:
    print(f'  - {comp["name"]} ({comp["type"]})')
    print(f'    Responsibility: {comp["responsibility"]}')
    print(f'    Modules: {len(comp["modules"])}')

print(f'\nWorkflows ({len(arch_context["workflows"])} total):')
for wf in arch_context['workflows']:
    print(f'  - {wf["name"]}: {wf["criticality"]}')
    print(f'    Entry: {wf["entry"]}')

print(f'\nTechnologies:')
for key, value in arch_context['technologies'].items():
    print(f'  {key}: {value}')

print(f'\nCritical Integrations:')
for integration in arch_context['critical_integrations']:
    print(f'  - {integration["name"]} ({integration["type"]})')

print(f'\nConstraints:')
constraints = arch_context['constraints']
print(f'  Max HLD containers: {constraints["max_hld_containers"]}')
print(f'  Max LLD classes: {constraints["max_lld_class_diagram_classes"]}')
print(f'  Must-include layers: {constraints["must_include_layers"]}')
print(f'  Must-highlight workflows: {constraints["must_highlight_workflows"]}')

print('\n✓ Phase 1 extraction complete!')
