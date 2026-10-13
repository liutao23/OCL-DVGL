"""Resolve local University1652 and height-specific SUES200 datasets."""
from pathlib import Path


def dataset_path(root, dataset='U1652', datasets_root=None):
    aliases = {'University1652': 'U1652', 'university1652': 'U1652'}
    name = aliases.get(dataset, dataset)
    if name.startswith('SUES'):
        name = name[4:]
    if name not in ('U1652', '150', '200', '250', '300'):
        raise ValueError(f'Unsupported dataset: {dataset}')
    root = Path(root).expanduser().resolve()
    if datasets_root is not None:
        base = Path(datasets_root).expanduser()
        base = base if base.is_absolute() else root / base
        return (base / name).resolve()
    for parent in (root, *root.parents):
        candidate = parent / 'Datasets' / name
        if candidate.is_dir():
            return candidate.resolve()
    return root / 'Datasets' / name


def add_dataset_arguments(parser):
    parser.add_argument('--dataset', default='U1652',
                        help='U1652/University1652 or SUES height: 150, 200, 250, 300 (SUES200 = height 200)')
    parser.add_argument('--datasets-root', type=Path,
                        help='Directory containing U1652, 150, 200, 250, 300; otherwise search root and ancestors')
