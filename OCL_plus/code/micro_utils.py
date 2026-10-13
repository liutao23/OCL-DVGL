"""Micro-only logging, AMP, and output protection."""
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import torch
from common import guard, dump


def amp(device):
    device = torch.device(device)
    return torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def output_guard(path, root, smoke):
    out = guard(path)
    for source in (guard(root/'Datasets'), guard(root/'3DGS重建'), guard(root/'OCL_plus'/'code'),
                   guard(root/'OCL_plus'/'models'), guard(smoke)):
        if out == source or source in out.parents or out in source.parents:
            raise ValueError(f'Output directory overlaps immutable source: {source}')
    return out


def save_torch(path, value):
    path = guard(path)
    temp = guard(path.with_suffix(path.suffix+'.tmp'))
    torch.save(value, temp)
    temp.replace(path)


def load_torch(path):
    # Only our user-selected local checkpoints, never remote/untrusted downloads.
    return torch.load(guard(path), map_location='cpu', weights_only=False)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with guard(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


