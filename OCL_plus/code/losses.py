"""Only partial/full cosine distillation and cross-place InfoNCE."""
import math
import torch
from torch.nn import functional as F


def objective(student, teacher, labels, temperature=.07, lambda_nce=1., lambda_distill=1.):
    if (not all(math.isfinite(w) and w >= 0 for w in (lambda_distill, lambda_nce))
            or lambda_distill + lambda_nce == 0):
        raise ValueError('Loss weights must be finite, nonnegative, and not both zero')
    teacher = teacher.detach().float()
    student = student.float()
    distill = (1-(student*teacher[labels]).sum(-1)).mean() if lambda_distill else student.new_zeros(())
    nce = student.new_zeros(())
    accuracy = 0.
    if lambda_nce:
        logits = student@teacher.T/temperature
        nce = F.cross_entropy(logits, labels)
        accuracy = float((logits.argmax(-1)==labels).float().mean())
    total = lambda_distill*distill+lambda_nce*nce
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite distillation/InfoNCE loss')
    return total, dict(total=float(total.detach()), distill=float(distill.detach()),
                       nce=float(nce.detach()), nce_accuracy=accuracy)


def ema_momentum(step, total, start=.996, end=.9999):
    p = step/max(total-1, 1)
    return end-(end-start)*(1+math.cos(math.pi*p))/2


@torch.no_grad()
def update_ema(teacher, student, momentum):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(momentum).add_(s.detach(), alpha=1-momentum)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.copy_(s)
    teacher.eval()
