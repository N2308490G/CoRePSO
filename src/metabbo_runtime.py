"""Load selected pinned MetaBox modules without unrelated benchmark imports."""
import importlib
import sys
import types
from pathlib import Path


def load_method(name):
    if name not in ('DEDQN', 'RLHPSDE', 'RLEPSO', 'GLEET', 'SYMBOL'):
        raise ValueError(name)
    root = Path(__file__).resolve().parents[1] / 'baselines/MetaBox/src'
    # Package namespaces retain upstream relative imports and pickle identities.
    for suffix in ('', '.baseline', '.baseline.metabbo', '.environment',
                   '.environment.problem', '.environment.optimizer', '.rl'):
        key = 'corepso_metabox' + suffix
        if key not in sys.modules:
            package = types.ModuleType(key)
            package.__path__ = [str(root.joinpath(*suffix.strip('.').split('.'))) if suffix else str(root)]
            sys.modules[key] = package
    agent = importlib.import_module('corepso_metabox.baseline.metabbo.' + name.lower())
    optimizer = importlib.import_module('corepso_metabox.environment.optimizer.' + name.lower() + '_optimizer')
    return getattr(agent, name), getattr(optimizer, name + '_Optimizer')
