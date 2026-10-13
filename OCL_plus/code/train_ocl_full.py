#!/usr/bin/env python3
"""Source-only full OCL: logical place batches, global prototype negatives, rotating memory."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import guard, dump
from dino_backbone import DinoBackbone
from orbitformer import OCLModel
from full_data import FullOrbitData, digest
from full_memory import FullMemory
from full_training_utils import optimizer_for, lr_multiplier, capture_rng, restore_rng, source_metrics
from ocl_data import locate_checkpoint, sha256
from losses import objective, update_ema, ema_momentum
from micro_utils import amp, output_guard, save_torch, load_torch, write_csv
from checkpoint_selection import select_checkpoint_source_retrieval


def replace_with_hardlink(source, target):
    """Atomically reuse identical checkpoint bytes when best is also last."""
    temp=target.with_suffix(target.suffix+'.tmp')
    try:
        if temp.exists():temp.unlink()
        os.link(source,temp)
        temp.replace(target)
        return True
    except OSError:
        if temp.exists():temp.unlink()
        return False


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--orbit-root',type=Path,default=Path('OCL_plus/outputs/orbit_full'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('OCL_plus/outputs/train_full_ocl'))
    p.set_defaults(variant='mean_teacher', lambda_distill=1., lambda_nce=1.)
    p.add_argument('--device',default='cuda:0')
    for name,default in [('seed',0),('epochs',20),('steps-per-epoch',100),('batch-places',8),
                         ('place-microbatch',2),('dino-chunk-size',2),('prototype-batch-places',8)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('memory-refresh-fraction',.20),('memory-momentum',.9),('backbone-lr',1e-6),
                         ('new-module-lr',1e-4),('weight-decay',.05),('layer-decay',.80),('warmup-fraction',.05),
                         ('temperature',.07),('ema-start',.996),('ema-end',.9999)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--split-seed',type=int,default=None,help='Fixed development split seed; defaults to training seed')
    p.add_argument('--loader-workers',type=int,default=8,
                   help='Parallel per-place image decode/preprocess workers; default: 8')
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=False,
                   help='Trade speed for lower training memory; optimized default: disabled')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--final-refit',action='store_true')
    p.add_argument('--fixed-epochs',type=int)
    a = p.parse_args()
    if a.final_refit:
        if not a.fixed_epochs or a.fixed_epochs<1:
            p.error('--final-refit requires positive --fixed-epochs E')
        a.epochs = a.fixed_epochs
    elif a.fixed_epochs is not None:
        p.error('--fixed-epochs requires --final-refit')
    if any(getattr(a,k)<1 for k in ('epochs','steps_per_epoch','batch_places','place_microbatch','dino_chunk_size','prototype_batch_places')):
        p.error('Counts must be positive')
    if a.loader_workers<0:p.error('--loader-workers must be nonnegative')
    if not 0<a.memory_refresh_fraction<=1 or not 0<=a.memory_momentum<1 or not 0<=a.warmup_fraction<1:
        p.error('Invalid refresh/momentum/warmup fraction')
    if not 0<=a.ema_start<=a.ema_end<1 or a.temperature<=0 or a.lambda_nce<0:
        p.error('Invalid loss/EMA settings')
    a.root,a.orbit_root = guard(a.root),guard(a.orbit_root)
    a.output_dir = output_guard(a.output_dir,a.root,a.orbit_root)
    a.checkpoint = locate_checkpoint(a.root,a.checkpoint)
    return a


@torch.no_grad()
def validate_source(student, data, bank, a, epoch):
    student.eval()
    tasks = {}
    validation_data=copy.copy(data)
    validation_data.paths=data.all_paths
    pools = ('real',)
    for pool in pools:
        queries = [(sid,[i]) for sid in data.val for i in data.validation_indices(sid, pool)]
        descriptors, labels = [], []
        label_map = {sid:i for i,sid in enumerate(data.val)}
        io_batch=max(a.dino_chunk_size,a.dino_chunk_size*max(1,a.loader_workers))
        for start in range(0,len(queries),io_batch):
            batch = queries[start:start+io_batch]
            image_groups=validation_data.load_selections(batch,a.loader_workers)
            if torch.device(a.device).type=='cuda':image_groups=[x.pin_memory() for x in image_groups]
            with amp(a.device):
                if hasattr(student,'encode_preloaded'):
                    z = student.encode_preloaded(image_groups,a.device,a.dino_chunk_size)
                else:
                    z = student.encode(validation_data,batch,a.device,a.dino_chunk_size)
            descriptors.append(z.float())
            labels.extend(label_map[sid] for sid,_ in batch)
        tasks[pool] = source_metrics(torch.cat(descriptors),bank,labels)
    student.train()
    return dict(source_only=True,validation_place_count=len(data.val),epoch=epoch,tasks=tasks)


def export_image_model(path, student, config, epoch=None, selection=None):
    # Pure-image evaluator loads ONLY this export, never a teacher/memory training checkpoint.
    save_torch(path,dict(format='ocl_image_only_v1',student=student.state_dict(),
               variant=config['variant'],architecture='DINOv3_B16_ViewTokenizer_OrbitFormer',
               preprocessing=dict(resolution=256),config_fingerprint=config['config_fingerprint'],
               epoch=epoch,source_selection_score=selection,source_split_hash=config.get('split_hash'),
               dino_sha256=config.get('dino_sha256'),architecture_config=dict(
                   backbone='DINOv3_ViT_B16_LVD1689M',resolution=256,dim=768,view_queries=4,
                   view_depth=2,orbit_queries=9,orbit_depth=3,heads=8,mlp_ratio=4)))


def main():
    a = arguments()
    # Keep programmatic/test callers written before these performance options compatible.
    if not hasattr(a,'loader_workers'):a.loader_workers=0
    if not hasattr(a,'gradient_checkpointing'):a.gradient_checkpointing=True
    if not hasattr(a,'lambda_distill'):a.lambda_distill=1.
    started = time.perf_counter()
    split_options = {} if getattr(a, "split_seed", None) is None else {"split_seed": a.split_seed}
    data = FullOrbitData(a.root,a.orbit_root,a.variant,a.seed,a.final_refit,**split_options)
    if a.batch_places>len(data.train):
        raise ValueError('batch-places exceeds distinct training places')
    cfg = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'}
    cfg.update(schema='ocl_full_v1',dino_sha256=sha256(a.checkpoint),orbit_manifest_hash=data.orbit_hash,
        input_manifest_hash=data.input_hash,split_hash=data.split_hash,split=data.split,
        satellite_pixels_accessed=False,loss=('NCE_only' if a.lambda_distill==0 else
            'distill_only' if a.lambda_nce==0 else 'distill_plus_NCE'),
        scheduler='cosine_5pct_warmup',torch_version=torch.__version__)
    # Code changes affecting training cannot silently reuse old optimizer/memory state.
    names=['train_ocl_full.py','full_data.py','full_memory.py','full_training_utils.py','orbitformer.py',
           'view_tokenizer.py','dino_backbone.py','losses.py','orbit_memory.py','ocl_data.py','checkpoint_selection.py',
           'common.py','micro_utils.py']
    cfg['code_hash'] = digest({name:sha256(Path(__file__).with_name(name)) for name in names})
    cfg['config_fingerprint'] = digest(cfg)
    out = a.output_dir
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise FileExistsError('Output is not empty; use --resume or a new directory')
    config_path = out/'config.json'
    resume_fingerprint=None
    if a.resume:
        if not config_path.is_file():raise ValueError('Resume requires config.json')
        saved_cfg=json.loads(config_path.read_text())
        if saved_cfg['config_fingerprint']!=cfg['config_fingerprint']:
            ignored={'code_hash','config_fingerprint'}
            old_semantic={k:v for k,v in saved_cfg.items() if k not in ignored}
            new_semantic={k:v for k,v in cfg.items() if k not in ignored}
            old_semantic.setdefault('lambda_distill',1.)
            if old_semantic!=new_semantic:
                raise ValueError('Resume configuration/data differs; use a new directory')
            resume_fingerprint=saved_cfg['config_fingerprint']
            print('Resuming across a code-only fingerprint change; training/data settings are identical.',flush=True)
    for sub in ('checkpoints','diagnostics','reports','memory/initial'):
        (out/sub).mkdir(parents=True,exist_ok=True)
    dump(config_path,cfg)
    if resume_fingerprint:
        dump(out/'reports/resume_migration.json',dict(
            reason='code_only_bugfix_or_performance_change',
            previous_config_fingerprint=resume_fingerprint,
            current_config_fingerprint=cfg['config_fingerprint']))
    dump(out/'split.json',data.split)
    if not a.final_refit:
        split_dir = guard(a.root/'OCL_plus/splits')
        split_dir.mkdir(parents=True,exist_ok=True)
        for name,ids in [('train',data.train),('val',data.val)]:
            (split_dir/f'full_dev_{name}.txt').write_text('\n'.join(ids)+'\n')
    if torch.device(a.device).type=='cuda':
        torch.cuda.set_device(a.device)
        torch.cuda.reset_peak_memory_stats(a.device)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(a.seed)
    student=OCLModel(DinoBackbone(a.checkpoint)).to(a.device)
    if hasattr(student,'gradient_checkpointing'):
        student.gradient_checkpointing=a.gradient_checkpointing
    teacher=copy.deepcopy(student).eval().requires_grad_(False)
    if hasattr(teacher,'gradient_checkpointing'):teacher.gradient_checkpointing=False
    data.transform=student.backbone.transform
    data.loader_workers=a.loader_workers
    memory_data=data.teacher_memory_data() if hasattr(data,'teacher_memory_data') else data
    optimizer=optimizer_for(student,a)
    total=a.epochs*a.steps_per_epoch
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_multiplier(step,total,a.warmup_fraction))
    scaler=torch.amp.GradScaler('cuda',enabled=torch.device(a.device).type=='cuda')
    memory=FullMemory(a.memory_momentum)
    history={'train':[],'validation':[]}
    best=None; epoch=0; step=0; elapsed_before=0.; initialized=False
    last=out/'checkpoints/last.pt'
    legacy_latest=out/'checkpoints/latest.pt'
    resume_checkpoint=last if last.is_file() else legacy_latest
    if a.resume and resume_checkpoint.is_file():
        state=load_torch(resume_checkpoint)
        if state['config_fingerprint'] not in {cfg['config_fingerprint'],resume_fingerprint}:
            raise ValueError('Checkpoint config fingerprint differs')
        student.load_state_dict(state['student']);teacher.load_state_dict(state['teacher'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);memory.load_state_dict(state['memory'])
        epoch,step,best,history=state['epoch'],state['step'],state['best'],state['history']
        initialized=state['initialized'];elapsed_before=state['elapsed_seconds']
        restore_rng(state['rng'],rng)
        del state
    def state_dict():
        return dict(config=cfg,config_fingerprint=cfg['config_fingerprint'],student=student.state_dict(),
            teacher=teacher.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),memory=memory.state_dict(),epoch=epoch,step=step,step_in_epoch=0,
            rng=capture_rng(rng),best=best,history=history,initialized=initialized,
            elapsed_seconds=elapsed_before+time.perf_counter()-started)
    if not initialized:
        # Freeze initial model/RNG state before encoding so interrupted initialization reuses per-place files.
        if not last.exists():
            save_torch(last,state_dict())
        memory.initialize(teacher,memory_data,a.device,a.dino_chunk_size,out/'memory/initial',cfg['config_fingerprint'])
        initialized=True
        save_torch(last,state_dict())
    labels={sid:i for i,sid in enumerate(data.train)}
    for current_epoch in range(epoch+1,a.epochs+1):
        memory.fractional_refresh(teacher,memory_data,a.device,a.dino_chunk_size,a.memory_refresh_fraction,current_epoch)
        bank=memory.bank(teacher,data.train,a.device,a.prototype_batch_places,a.variant,a.dino_chunk_size)
        val_bank=memory.bank(teacher,data.val,a.device,a.prototype_batch_places,a.variant,a.dino_chunk_size)
        student.train()
        for within in range(a.steps_per_epoch):
            ids=rng.choice(data.train,a.batch_places,replace=False).tolist()
            selections=data.selections(ids,rng,step/max(total-1,1))
            image_groups=data.load_selections(selections,a.loader_workers)
            if torch.device(a.device).type=='cuda':
                image_groups=[images.pin_memory() for images in image_groups]
            memory_selections,memory_images=(data.teacher_memory_inputs(selections,image_groups)
                if hasattr(data,'teacher_memory_inputs') else (selections,image_groups))
            memory.update_preloaded(teacher,memory_selections,memory_images,a.device,a.dino_chunk_size)
            optimizer.zero_grad(set_to_none=True)
            averaged=dict(total=0.,distill=0.,nce=0.,nce_accuracy=0.)
            for start in range(0,len(selections),a.place_microbatch):
                batch=selections[start:start+a.place_microbatch]
                batch_images=image_groups[start:start+a.place_microbatch]
                y=torch.tensor([labels[sid] for sid,_ in batch],device=a.device)
                with amp(a.device):
                    if hasattr(student,'encode_preloaded'):
                        z=student.encode_preloaded(batch_images,a.device,a.dino_chunk_size)
                    else:
                        z=student.encode(data,batch,a.device,a.dino_chunk_size)
                loss,metrics=objective(z,bank,y,a.temperature,a.lambda_nce,a.lambda_distill)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite full-scale loss')
                weight=len(batch)/len(selections)
                scaler.scale(loss*weight).backward()
                for k in averaged:
                    averaged[k]+=metrics[k]*weight
                del z,loss
            scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
            if not torch.isfinite(grad) and not scaler.is_enabled():
                raise FloatingPointError('Non-finite gradients without AMP scaler')
            scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            skipped=scaler.get_scale()<scale
            if not skipped:
                update_ema(teacher,student,ema_momentum(step,total,a.ema_start,a.ema_end))
                scheduler.step()
            step+=1
            history['train'].append(dict(epoch=current_epoch,step=step,**averaged,
                gradient_norm=float(grad) if torch.isfinite(grad) else None,amp_skipped=skipped,
                max_lr=max(g['lr'] for g in optimizer.param_groups),
                m1_count=sum(len(indices)==1 for _,indices in selections),
                place_count=len(ids),observations=sum(len(i) for _,i in selections)))
            print(f'epoch {current_epoch}/{a.epochs} step {within+1}/{a.steps_per_epoch} '
                  f'loss={averaged["total"]:.5f} NCE={averaged["nce"]:.5f}',flush=True)
        epoch=current_epoch
        improved=False
        if data.val:
            candidate=validate_source(student,data,val_bank,a,epoch)
            history['validation'].append(candidate)
            chosen=select_checkpoint_source_retrieval([candidate] if best is None else [best,candidate],
                preregistered_tasks=list(candidate['tasks']),min_val_places=min(20,len(data.val)))
            improved=best is None or chosen['epoch']!=best['epoch']
            best=chosen
        snapshot=state_dict()
        if improved:
            save_torch(out/'checkpoints/best.pt',snapshot)
            export_image_model(out/'checkpoints/best_image.pt',student,cfg,epoch,best)
        # Commit last epoch last: a crash cannot advertise incomplete progress.
        if not improved or not replace_with_hardlink(out/'checkpoints/best.pt',last):
            save_torch(last,snapshot)
        last_image=out/'checkpoints/last_image.pt'
        if not improved or not replace_with_hardlink(out/'checkpoints/best_image.pt',last_image):
            export_image_model(last_image,student,cfg,epoch,None)
        # New policy retains only best and last; remove legacy/per-epoch duplicates.
        for obsolete in (legacy_latest,out/'checkpoints/final.pt',out/'checkpoints/final_image.pt'):
            if obsolete.is_file():obsolete.unlink()
        for obsolete in (out/'checkpoints').glob('epoch_*.pt'):
            obsolete.unlink()
        del snapshot
        write_csv(out/'diagnostics/train.csv',history['train'])
        dump(out/'diagnostics/validation.json',history['validation'])
        dump(out/'memory/refresh_history.json',memory.refresh_log)
        dump(out/'reports/full_report.json',dict(status='complete' if epoch==a.epochs else 'running',
            epoch=epoch,step=step,best=best,config_fingerprint=cfg['config_fingerprint'],split=data.split,
            variant=a.variant,dino_sha256=cfg['dino_sha256'],orbit_manifest_hash=data.orbit_hash,
            split_hash=data.split_hash,memory_refresh_fraction=a.memory_refresh_fraction,
            runtime_seconds=elapsed_before+time.perf_counter()-started,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(a.device) if torch.device(a.device).type=='cuda' else 0,
            satellite_pixels_accessed=False))
        del bank,val_bank
    print(f'Best validation epoch: {best["epoch"] if best else "none"}',flush=True)
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':
    main()
