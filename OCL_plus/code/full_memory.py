"""Persistent initialization and fractional refresh built on OrbitMemory encoding/update."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from orbit_memory import OrbitMemory
from micro_utils import amp, save_torch, load_torch
from common import dump


def rotating_shard(total, fraction, epoch):
    if total<1 or not 0<fraction<=1 or epoch<1:
        raise ValueError('Invalid refresh schedule')
    count = int(math.ceil(1/fraction))
    return np.array_split(np.arange(total), count)[(epoch-1)%count].tolist()


class FullMemory(OrbitMemory):
    def initialize(self, teacher, data, device, chunk, folder, fingerprint):
        folder.mkdir(parents=True, exist_ok=True)
        teacher.eval()
        for n, sid in enumerate(data.ids):
            path = folder/f'{sid}.pt'
            if path.exists():
                cached = load_torch(path)
                if cached['fingerprint']!=fingerprint:
                    raise ValueError('Initialization memory fingerprint differs')
                value = cached['tokens']
                if value.shape != (len(data.paths[sid]),4,768) or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid initial memory {sid}')
                self.values[sid] = value
                continue
            indices = list(range(len(data.paths[sid])))
            value = self.compact(self._tokens(teacher, data, sid, indices, device, chunk))
            self.values[sid] = value
            save_torch(path, dict(fingerprint=fingerprint, tokens=value))
            print(f'Memory initialization {n+1}/{len(data.ids)}: {sid}', flush=True)
        dump(folder.parent/'manifest.json', dict(fingerprint=fingerprint, initialized_scene_ids=data.ids,
            observation_counts={sid:len(data.paths[sid]) for sid in data.ids}, shape_per_observation=[4,768],
            storage='CPU fp16 (fp32 fallback for overflow)', online_momentum=self.beta,
            initialization_files_immutable=True, training_memory_saved_in_checkpoint=True))

    @torch.no_grad()
    def fractional_refresh(self, teacher, data, device, chunk, fraction, epoch):
        teacher.eval()
        observations = [(sid,i) for sid in data.ids for i in range(len(data.paths[sid]))]
        selected = rotating_shard(len(observations), fraction, epoch)
        groups = {}
        for index in selected:
            sid, obs = observations[index]
            groups.setdefault(sid, []).append(obs)
        for sid, indices in groups.items():
            new = self._tokens(teacher, data, sid, indices, device, chunk)
            old = self.values[sid].float()
            old[indices] = new
            self.values[sid] = self.compact(old)
        row = dict(epoch=epoch, observations=len(selected), total=len(observations),
                   requested_fraction=fraction, shards=math.ceil(1/fraction))
        self.refresh_log.append(row)
        print(f'Memory rotating refresh: {len(selected)}/{len(observations)} observations', flush=True)
        return row

    @torch.no_grad()
    def bank(self, teacher, ids, device, place_chunk=8, variant='mean_teacher', image_chunk=4):
        teacher.eval()
        parts = []
        for start in range(0,len(ids),place_chunk):
            group = ids[start:start+place_chunk]
            means = []
            for sid in group:
                value = self.values[sid]
                descriptors = []
                for j in range(0,len(value),image_chunk):
                    with amp(device):
                        z = teacher.orbit(value[j:j+image_chunk].float().to(device)[:,None])
                    descriptors.append(z.float())
                means.append(F.normalize(torch.cat(descriptors).mean(0), dim=0))
            part = torch.stack(means)
            if not torch.isfinite(part).all():
                raise FloatingPointError('Non-finite teacher prototype bank')
            parts.append(part.detach().float())
        return torch.cat(parts) if parts else torch.empty(0,768,device=device)
