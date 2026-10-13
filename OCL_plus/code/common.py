"""Guarded source-only I/O and deterministic serialization."""
import ast
import json
import re
from pathlib import Path


def guard(path):
    p = Path(path).expanduser()
    for candidate in (p.absolute(), p.resolve()):
        if any(part.lower() == 'satellite' for part in candidate.parts):
            raise ValueError(f'Forbidden satellite path: {candidate}')
    return p.resolve()


def dump(path, value):
    p = guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(p)


def config(path):
    """Read Namespace literals without eval or executing config code."""
    p = guard(path)
    if not p.is_file():
        return {}
    node = ast.parse(p.read_text(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace' or node.args:
        raise ValueError(f'Unsupported config: {p}')
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}


def natural_key(value):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(value))]


def fatal_cuda_error(exc):
    """A poisoned CUDA context cannot safely process the next view/scene."""
    message = str(exc).lower()
    return any(term in message for term in (
        'illegal memory access', 'device-side assert', 'unspecified launch failure',
        'misaligned address', 'illegal instruction'))
