"""Teacher ViewTokenizer memory, indexed by scene and fixed observation index."""
import time
import torch
from micro_utils import amp, sync


class OrbitMemory:
    def __init__(self, beta=.9):
        self.beta, self.values, self.refresh_log = beta, {}, []

    @torch.no_grad()
    def _tokens(self, teacher, data, sid, indices, device, chunk):
        tokens = []
        image_chunks=data.image_chunks(sid,indices,chunk) if hasattr(data,'image_chunks') else [
            data.images(sid,indices[start:start+chunk]) for start in range(0,len(indices),chunk)]
        for images in image_chunks:
            if torch.device(device).type=='cuda':images=images.pin_memory()
            images = images.to(device,non_blocking=True)
            with amp(device):
                value = teacher.image_tokens(images)
            value = value.float().cpu()
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'Non-finite teacher memory: {sid}')
            tokens.append(value)
        return torch.cat(tokens)

    @staticmethod
    def compact(value):
        half = value.half()
        # Keep float32 if float16 would overflow; never normalize raw view tokens.
        return half if torch.isfinite(half).all() else value.float()

    @torch.no_grad()
    def refresh(self, teacher, data, device, chunk, ids=None, epoch=0, reason='epoch_start'):
        teacher.eval()
        sync(device)
        start = time.perf_counter()
        ids = ids or data.ids
        for sid in ids:
            self.values[sid] = self.compact(self._tokens(teacher, data, sid, list(range(len(data.paths[sid]))), device, chunk))
        sync(device)
        entry = dict(epoch=epoch, reason=reason, seconds=time.perf_counter()-start, scenes=list(ids))
        self.refresh_log.append(entry)
        print(f'Memory refresh {reason}: {len(ids)} scenes, {entry["seconds"]:.1f}s', flush=True)
        return entry

    @torch.no_grad()
    def update(self, teacher, data, selections, device, chunk):
        teacher.eval()
        for sid, indices in selections:
            values = self._tokens(teacher, data, sid, indices, device, chunk)
            old = self.values[sid].float()
            old[indices] = self.beta*old[indices]+(1-self.beta)*values
            self.values[sid] = self.compact(old)

    @torch.no_grad()
    def update_preloaded(self, teacher, selections, image_groups, device, chunk):
        """Update memory from the same transformed tensors consumed by the student."""
        teacher.eval()
        for (sid,indices),images in zip(selections,image_groups):
            parts=[]
            for start in range(0,len(images),chunk):
                with amp(device):
                    value=teacher.image_tokens(images[start:start+chunk].to(device,non_blocking=True))
                parts.append(value.float().cpu())
            values=torch.cat(parts)
            if not torch.isfinite(values).all():
                raise FloatingPointError(f'Non-finite teacher memory: {sid}')
            old=self.values[sid].float()
            old[indices]=self.beta*old[indices]+(1-self.beta)*values
            self.values[sid]=self.compact(old)

    @torch.no_grad()
    def targets(self, teacher, ids, device, return_norm=False):
        # OCL+ permits different source observation counts across scenes.
        outputs, norms = [], []
        for sid in ids:
            views = self.values[sid].float().to(device)[None]
            with amp(device):
                result = teacher.orbit(views, return_norm=return_norm)
            if return_norm:
                z, n = result; outputs.append(z); norms.append(n)
            else:
                outputs.append(result)
        if not outputs:
            empty = torch.empty(0, 768, device=device)
            return (empty, torch.empty(0, device=device)) if return_norm else empty
        z = torch.cat(outputs, 0)
        if not torch.isfinite(z).all():
            raise FloatingPointError('Non-finite full-orbit teacher prototypes')
        if return_norm:
            n = torch.cat(norms, 0)
            if not torch.isfinite(n).all():
                raise FloatingPointError('Non-finite teacher pre-L2 norms')
            return z, n
        return z

    def state_dict(self):
        return dict(beta=self.beta, values=self.values, refresh_log=self.refresh_log)

    def load_state_dict(self, state):
        self.beta, self.values = state['beta'], state['values']
        self.refresh_log = state['refresh_log']
