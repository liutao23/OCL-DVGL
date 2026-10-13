"""Shared optimizer and deterministic state for full-scale training."""
import math
import random
import numpy as np
import torch


def optimizer_for(student, args):
    groups = {}
    depth = len(student.backbone.model.blocks)
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            parts = name.split('.')
            layer = int(parts[parts.index('blocks')+1])+1 if 'blocks' in parts else (depth+1 if '.norm.' in name else 0)
            lr = args.backbone_lr*args.layer_decay**(depth+1-layer)
        else:
            layer, lr = -1, args.new_module_lr
        decay = args.weight_decay if param.ndim>1 and not any(k in name for k in ('queries','cls_token','reg_token')) else 0.
        groups.setdefault((layer,lr,decay), dict(params=[],lr=lr,weight_decay=decay))['params'].append(param)
    return torch.optim.AdamW(list(groups.values()))


def lr_multiplier(step, total, warmup_fraction=.05):
    warmup = max(1, int(math.ceil(total*warmup_fraction)))
    if step<warmup:
        return (step+1)/warmup
    progress = min(1., (step-warmup)/max(1,total-warmup-1))
    return .5*(1+math.cos(math.pi*progress))


def capture_rng(rng):
    return dict(python=random.getstate(), numpy=np.random.get_state(), generator=rng.bit_generator.state,
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state, rng):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    rng.bit_generator.state = state['generator']
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def source_metrics(queries, bank, labels):
    scores = queries.float()@bank.float().T
    if not torch.isfinite(scores).all() or bank.shape[0]<2:
        raise ValueError('Source metrics require finite scores and at least two prototypes')
    labels = torch.as_tensor(labels,device=scores.device)
    cols = torch.arange(bank.shape[0],device=scores.device)[None]
    positive = scores.gather(1,labels[:,None]).squeeze(1)
    rank = ((scores>positive[:,None])|((scores==positive[:,None])&(cols<labels[:,None]))).sum(1)+1
    negative = scores.masked_fill(cols==labels[:,None],0).sum(1)/(bank.shape[0]-1)
    hard = scores.masked_fill(cols==labels[:,None],-torch.inf).max(1).values
    rr = float(rank.float().reciprocal().mean())
    return dict(r1=float((rank<=1).float().mean()), r5=float((rank<=5).float().mean()),
                r10=float((rank<=10).float().mean()), mrr=rr, map=rr,
                mean_margin=float((positive-negative).mean()), hard_margin=float((positive-hard).mean()),
                query_count=len(labels), prototype_count=bank.shape[0])
