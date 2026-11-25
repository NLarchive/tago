"""Hexagonal ring rule: inner rings never import outer rings (checked on the source AST)."""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / 'tago'
ALLOWED = {  # ring of the importing module -> rings it may import
    'core.domain': {'core.domain'},
    'core.ports': {'core.domain', 'core.ports'},
    'infrastructure': {'core.domain', 'core.ports', 'infrastructure', 'config'},
    'application': {'core.domain', 'core.ports', 'application', 'config'},
    'workflows': {'core.domain', 'core.ports', 'infrastructure', 'application', 'workflows', 'config'},
    'gateway': {'core.domain', 'core.ports', 'infrastructure', 'application', 'workflows', 'gateway', 'config'},
}

def ring(module):
    parts = module.split('.')[1:]
    if not parts: return 'root'
    if parts[0] == 'core': return 'core.' + parts[1] if len(parts) > 1 else 'core'
    return parts[0]

def test_no_inner_ring_imports_outer_ring():
    violations = []
    for path in PACKAGE.rglob('*.py'):
        rel = path.relative_to(PACKAGE.parent).with_suffix('').as_posix().replace('/', '.')
        if rel.endswith('__init__') or rel.endswith('__main__') or rel == 'tago.config': continue
        source = ring(rel)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('tago'):
                target = ring(node.module)
                if target not in ALLOWED[source]: violations.append(f'{rel} -> {node.module}')
    assert not violations, violations
