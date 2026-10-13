"""Frozen geometry assets and source-only sampling. No geometry generation."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
from common import guard, natural_key
CHECKPOINT_NAME = 'DINOv3_ViTB16_LVD1689M.safetensors'
CURRICULUM = [dict(until=0.25, m=[16, 8, 4, 2, 1], p=[0.4, 0.35, 0.15, 0.05, 0.05]), dict(until=0.6, m=[8, 4, 2, 1], p=[0.25, 0.3, 0.25, 0.2]), dict(until=1.01, m=[4, 2, 1], p=[0.15, 0.3, 0.55])]

def sha256(path):
    h = hashlib.sha256()
    with guard(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def locate_checkpoint(root, explicit=None):
    if explicit:
        p = guard(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    preferred = [root / 'OCL_plus' / 'models' / CHECKPOINT_NAME, root / 'OCL' / 'models' / CHECKPOINT_NAME, Path('/media/liutao/A/3SGS_colmap/论文实验/模型') / CHECKPOINT_NAME]
    for p in preferred:
        if guard(p).is_file():
            return guard(p)
    for base in (root, Path('/media/liutao/A')):
        for folder, dirs, files in os.walk(guard(base), followlinks=False):
            allowed = []
            for name in sorted(dirs):
                if name.lower() in {'satellite', '.git', 'node_modules', '__pycache__'}:
                    continue
                try:
                    guard(Path(folder) / name)
                    allowed.append(name)
                except ValueError:
                    pass
            dirs[:] = allowed
            if CHECKPOINT_NAME in files:
                return guard(Path(folder) / CHECKPOINT_NAME)
    raise FileNotFoundError(f'No local DINOv3 checkpoint. Place it at {preferred[0]}; no downloads are allowed.')

class OrbitData:

    def images(self, sid, indices):
        if self.transform is None:
            raise RuntimeError('DINO preprocessing was not attached')
        tensors = []
        for i in indices:
            with Image.open(guard(self.paths[sid][int(i)])) as image:
                tensors.append(self.transform(image.convert('RGB')))
        return torch.stack(tensors)

    def load_selections(self, selections, workers=0):
        """Decode/transform a logical batch once while preserving its order."""
        if workers <= 1:
            return [self.images(sid, indices) for sid, indices in selections]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.images, sid, indices) for sid, indices in selections]
            return [future.result() for future in futures]

    def image_chunks(self, sid, indices, chunk, workers=None):
        """Decode ordered chunks concurrently for memory initialization/refresh."""
        chunks = [indices[start:start + chunk] for start in range(0, len(indices), chunk)]
        workers = getattr(self, 'loader_workers', 0) if workers is None else workers
        if workers <= 1:
            return [self.images(sid, part) for part in chunks]
        with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            futures = [pool.submit(self.images, sid, part) for part in chunks]
            return [future.result() for future in futures]
