#!/usr/bin/env python3
"""Drone-to-satellite image retrieval using an OCL student or frozen DINO backbone."""
import argparse
import csv
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from common import guard, dump
from eval_ocl_support import (checkpoint_metadata, sha256, cache_identity, read_cache, write_cache,
                              write_csv)
from dino_backbone import DinoBackbone

SUFFIXES={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}


class EvaluationImageDataset(Dataset):
    """Parallel image loading/preprocessing while preserving entry order."""
    def __init__(self,entries,transform):self.entries,self.transform=entries,transform
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        path,_=self.entries[index]
        with Image.open(path) as image:return self.transform(image.convert('RGB'))


def image_list(root, mapping=None):
    root=guard(root)
    if mapping:
        with guard(mapping).open(newline='') as f:
            rows=list(csv.DictReader(f))
        result=[]
        for row in rows:
            relative=Path(row['path'])
            path=guard(root/relative)
            if root not in path.parents or not path.is_file() or not row['place_id']:
                raise ValueError('Mapping requires existing relative image paths and nonempty place_id')
            result.append((path,row['place_id']))
    else:
        result=[]
        for folder,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d.lower()!='satellite' and not (Path(folder)/d).is_symlink())
            for name in sorted(files):
                raw=Path(folder)/name
                if raw.suffix.lower() not in SUFFIXES or raw.is_symlink():
                    continue
                path=guard(raw)
                relative=path.relative_to(root)
                if len(relative.parts)<2:
                    raise ValueError('Use <root>/<place_id>/<image> or supply --*-map CSV')
                result.append((path,relative.parts[0]))
    if not result or len({str(p) for p,_ in result})!=len(result):
        raise ValueError('Image list is empty or has duplicate paths')
    return result


@torch.inference_mode()
def encode(model, entries, transform, batch_size, device, frozen=False, description="Encoding images",
           encoder_mode=None, workers=0, prefetch_factor=2):
    result=[]
    options=dict(batch_size=batch_size,shuffle=False,num_workers=workers,
                 pin_memory=str(device).startswith('cuda'))
    if workers:
        options.update(persistent_workers=True,prefetch_factor=prefetch_factor)
    loader=DataLoader(EvaluationImageDataset(entries,transform),**options)
    with tqdm(total=len(entries), desc=description, unit='image', dynamic_ncols=True) as progress:
        for x in loader:
            x=x.to(device,non_blocking=True)
            z=model.orbit(model.image_tokens(x)[:,None])
            if not torch.isfinite(z).all():
                raise FloatingPointError('Nonfinite image descriptor')
            result.append(z.float().cpu())
            progress.update(len(x))
    return torch.cat(result)


def retrieval_metrics(q,g,queries,gallery):
    """Original evaluator's ranking/AP loop, shared unchanged across all encoders."""
    rows=[];missing=[];per_query=[];one_positive=True
    for i,(path,label) in enumerate(tqdm(queries, desc="Retrieval metrics", unit="query", dynamic_ncols=True)):
        eligible=torch.tensor([str(p)!=str(path) for p,_ in gallery])
        relevant=torch.tensor([place==label for _,place in gallery]) & eligible
        if not relevant.any():
            missing.append(str(path));continue
        scores=q[i]@g.T
        indices=torch.arange(len(gallery))[eligible]
        order=indices[torch.argsort(scores[eligible],descending=True,stable=True)]
        hits=relevant[order]
        ranks=torch.nonzero(hits).flatten()+1
        first=int(ranks[0]);ap=float((torch.arange(1,len(ranks)+1)/ranks.float()).mean())
        rows.append(dict(query=str(path),place_id=label,r1=float(first<=1),r5=float(first<=5),
                         r10=float(first<=10),mrr=1/first,ap=ap,top1=str(gallery[int(order[0])][0])))
        one_positive=one_positive and int(relevant.sum())==1
        positive=float(scores[relevant].max())
        negatives=eligible & ~relevant
        hardest=float(scores[negatives].max()) if negatives.any() else None
        per_query.append(dict(query_path=str(path),query_place_id=label,correct_gallery_place_id=label,
            correct_rank=first,reciprocal_rank=1/first,hit_at_1=float(first<=1),hit_at_5=float(first<=5),
            hit_at_10=float(first<=10),top1_gallery_place_id=gallery[int(order[0])][1],
            top1_similarity=float(scores[order[0]]),positive_similarity=positive,
            hardest_negative_similarity=hardest,hard_margin=positive-hardest if hardest is not None else None,
            average_precision=ap,relevant_gallery_count=int(relevant.sum())))
    if not rows:
        raise ValueError('No query has a relevant gallery image after self-match exclusion')
    metrics={key:sum(r[key] for r in rows)/len(rows) for key in ('r1','r5','r10','mrr','ap')}
    metrics['map']=metrics.pop('ap')
    if one_positive and not abs(metrics['mrr']-metrics['map'])<1e-6:
        raise ValueError(f'One-positive MRR/mAP discrepancy: {metrics}; check label/ranking semantics')
    return metrics,rows,missing,per_query


def fixed_protocol(queries,gallery,direction='drone_to_satellite'):
    if direction != 'drone_to_satellite':
        raise ValueError('Unknown retrieval direction')
    expected=(37855,951)
    if (len(queries),len(gallery))!=expected:
        raise ValueError(f'{direction} expects {expected[0]}/{expected[1]} images, got {len(queries)}/{len(gallery)}')
    from collections import Counter
    counts=Counter(label for _,label in gallery)
    bad=sorted({label for _,label in queries if counts[label]!=1})
    if bad:
        raise ValueError(f'Query/gallery positive-label discrepancy for {direction}: {bad}')



def load_encoder(a):
    mode='ocl_image_model'
    if not a.checkpoint:
        raise ValueError('OCL mode requires an explicit --checkpoint')
    state=torch.load(guard(a.checkpoint),map_location='cpu',weights_only=True)
    meta=checkpoint_metadata(a.checkpoint,state,getattr(a,'expected_variant',None))
    from orbitformer import OCLModel
    model=OCLModel(DinoBackbone())
    model.load_state_dict(state['student'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise ValueError('Nonfinite trained state')
    transform=model.backbone.transform;preprocess=model.backbone.preprocess_config
    del state
    if getattr(a,'dino_checkpoint',None):
        supplied=sha256(a.dino_checkpoint)
        if meta['dino_sha256'] is not None and meta['dino_sha256']!=supplied:
            raise ValueError('DINO initialization hash differs from explicitly supplied common backbone')
        meta['provided_dino_sha256']=supplied
        if meta['dino_sha256'] is None:
            # Unknown provenance remains explicit; never pretend a supplied file proves training history.
            raise ValueError('Export lacks verifiable DINO hash; retain matching source config or re-export metadata')
    import timm
    preprocess=dict(preprocess,transform_repr=repr(transform),precision='float32',timm_version=timm.__version__)
    return model.eval().requires_grad_(False),transform,meta,mode,preprocess


def run_evaluation(a):
    if a.batch_size<1:raise ValueError('batch-size must be positive')
    if getattr(a,'direction','drone_to_satellite') != 'drone_to_satellite':
        raise ValueError('Only drone_to_satellite is supported')
    print('Reading query/gallery file lists...',flush=True)
    queries=image_list(a.query_dir,getattr(a,'query_map',None))
    gallery=image_list(a.gallery_dir,getattr(a,'gallery_map',None))
    out=guard(a.output_dir)
    for source in (guard(a.query_dir),guard(a.gallery_dir)):
        if out==source or source in out.parents or out in source.parents:
            raise ValueError('Output must be separate from image roots')
    print('Loading image model...',flush=True)
    model,transform,meta,mode,preprocess=load_encoder(a)
    out.mkdir(parents=True,exist_ok=True)
    variant=getattr(a,'variant_name',None) or meta['stored_variant']
    if any(ch in variant for ch in '/\\') or variant in ('.','..'):
        raise ValueError('Invalid cache variant name')
    default_cache=Path(__file__).resolve().parents[1]/'outputs/test_eval/cache'
    cache=guard(getattr(a,'cache_root',None) or default_cache)/variant/meta['sha256']
    for source in (guard(a.query_dir),guard(a.gallery_dir)):
        if cache==source or source in cache.parents or cache in source.parents:
            raise ValueError('Cache overlaps image inputs')
    identity=cache_identity(meta,mode,preprocess,queries,gallery)
    cached=None if getattr(a,'no_cache',False) else read_cache(cache,identity,queries,gallery)
    if cached is None:
        model.to(a.device)
        print(f'Query: {len(queries):,}; gallery: {len(gallery):,}',flush=True)
        workers=getattr(a,'workers',8);prefetch=getattr(a,'prefetch_factor',2)
        q=encode(model,queries,transform,a.batch_size,a.device,False,'Query encoding',mode,workers,prefetch)
        g=encode(model,gallery,transform,a.batch_size,a.device,False,'Gallery encoding',mode,workers,prefetch)
        write_cache(cache,identity,q,g,queries,gallery)
    else:
        q,g=cached
    del model
    if str(a.device).startswith('cuda') and torch.cuda.is_available():torch.cuda.empty_cache()
    metrics,rows,missing,per_query=retrieval_metrics(q,g,queries,gallery)
    dump(out/'retrieval.json',dict(metrics=metrics,evaluated_queries=len(rows),gallery_images=len(gallery),
         queries_without_positive=missing,results=rows,inference=mode if mode!='ocl_image_model' else
         'raw_image->DINO_patches->ViewTokenizer->OrbitFormer(m=1)->L2_768',self_matches_excluded=True))
    # Retain the old output schema/artifact for existing consumers.
    torch.save(dict(query=q,gallery=g,query_paths=[str(p) for p,_ in queries],
                    gallery_paths=[str(p) for p,_ in gallery]),out/'descriptors.pt')
    write_csv(out/'per_query.csv',per_query)
    result=dict(variant=variant,direction='drone_to_satellite',metrics=metrics,query_count=len(queries),evaluated_queries=len(rows),gallery_count=len(gallery),
                checkpoint=meta['path'],checkpoint_sha=meta['sha256'],checkpoint_metadata=meta,encoder_mode=mode,
                cache_location=str(cache),query_file_list_hash=identity['query_file_list_hash'],
                gallery_file_list_hash=identity['gallery_file_list_hash'],preprocessing_fingerprint=identity['preprocessing_fingerprint'],
                metric_protocol='original_stable_cosine_rank_AP_v1')
    dump(out/'metrics.json',result)
    print(json.dumps(metrics,indent=2),flush=True)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,help='Explicit image-only OCL student export')
    p.add_argument('--encoder-mode',choices=('ocl_image_model',))
    p.add_argument('--expected-variant',choices=('mean_teacher',))
    p.add_argument('--query-dir',type=Path,required=True,help='Drone query image directory')
    p.add_argument('--gallery-dir',type=Path,required=True,help='Satellite gallery image directory')
    p.add_argument('--query-map',type=Path)
    p.add_argument('--gallery-map',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--no-cache',action='store_true',help='Re-encode instead of reusing a committed cache')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=8,help='Parallel image decode/preprocess workers; default: 8')
    p.add_argument('--prefetch-factor',type=int,default=2,help='Batches queued per worker; default: 2')
    return p


def main():
    p=parser();a=p.parse_args()
    if a.batch_size<1 or a.workers<0 or a.prefetch_factor<1:
        p.error('batch-size/prefetch-factor must be positive and workers nonnegative')
    run_evaluation(a)


if __name__=='__main__':main()
