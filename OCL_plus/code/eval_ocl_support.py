"""Checkpoint provenance and descriptor-cache I/O; no geometry or training imports."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from common import guard, dump

ARCHITECTURE='DINOv3_B16_ViewTokenizer_OrbitFormer'
ARCH_CONFIG=dict(backbone='DINOv3_ViT_B16_LVD1689M',resolution=256,dim=768,view_queries=4,
                 view_depth=2,orbit_queries=9,orbit_depth=3,heads=8,mlp_ratio=4)
def sha256(path):
    h=hashlib.sha256()
    with guard(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def file_list_hash(entries):
    return digest([(str(p),label,p.stat().st_size,p.stat().st_mtime_ns) for p,label in entries])


def checkpoint_metadata(path,state,expected_variant=None):
    path=guard(path)
    if state.get('format')!='ocl_image_only_v1' or set(state)&{'teacher','memory','optimizer'}:
        raise ValueError('Expected image-only export; full training checkpoints are forbidden')
    variant=state.get('variant')
    if variant != 'mean_teacher':
        raise ValueError(f'Missing/unknown trained variant: {variant}')
    if expected_variant and expected_variant!=variant:
        raise ValueError(f'CLI variant {expected_variant} != checkpoint variant {variant}')
    if state.get('architecture')!=ARCHITECTURE or state.get('preprocessing',{}).get('resolution')!=256:
        raise ValueError('Checkpoint architecture/resolution mismatch')
    if state.get('architecture_config') is not None and state['architecture_config']!=ARCH_CONFIG:
        raise ValueError('Detailed checkpoint architecture mismatch')
    if not isinstance(state.get('student'),dict) or not all(any(k.startswith(prefix) for k in state['student'])
            for prefix in ('backbone.','tokenizer.','orbit.')):
        raise ValueError('Missing trained backbone/tokenizer/orbit state keys')
    meta=dict(path=str(path),sha256=sha256(path),stored_variant=variant,
              source_split_hash=state.get('source_split_hash'),epoch=state.get('epoch'),
              source_selection_score=state.get('source_selection_score'),architecture=ARCHITECTURE,
              architecture_config=state.get('architecture_config',ARCH_CONFIG),
              dino_sha256=state.get('dino_sha256'),input_resolution=256,metadata_sources=['image_export'],
              config_fingerprint=state.get('config_fingerprint'))
    # Old exports omitted epoch/DINO/split. Read only associated JSON, never best.pt/teacher/memory.
    run=path.parent.parent;config=run/'config.json';report=run/'reports/full_report.json'
    if config.is_file():
        cfg=json.loads(config.read_text())
        if cfg.get('config_fingerprint')!=meta['config_fingerprint'] or cfg.get('variant')!=variant:
            raise ValueError('Adjacent config does not match image-export fingerprint/variant')
        for field,key in [('source_split_hash','split_hash'),('dino_sha256','dino_sha256')]:
            if meta[field] is not None and cfg.get(key)!=meta[field]:
                raise ValueError(f'Conflicting {field} provenance')
            meta[field]=cfg.get(key)
        meta['metadata_sources'].append(str(config))
        if report.is_file():
            source=json.loads(report.read_text())
            if source.get('config_fingerprint')!=meta['config_fingerprint']:
                raise ValueError('Source report/export fingerprint mismatch')
            if path.name=='best_image.pt':
                selection=source.get('best')
                source_epoch=selection.get('epoch') if selection else None
            elif path.name=='final_image.pt':
                selection=None;source_epoch=source.get('epoch')
            else:
                selection=None;source_epoch=None
            if meta['epoch'] is not None and source_epoch is not None and meta['epoch']!=source_epoch:
                raise ValueError('Source report/export epoch mismatch')
            if meta['epoch'] is None:meta['epoch']=source_epoch
            if meta['source_selection_score'] is None:meta['source_selection_score']=selection
            meta['metadata_sources'].append(str(report))
    meta['unavailable_metadata']=[k for k in ('source_split_hash','epoch','source_selection_score','dino_sha256') if meta[k] is None]
    return meta


def cache_identity(meta,mode,preprocess,queries,gallery):
    identity = dict(schema='ocl_descriptor_cache_v1',checkpoint_sha=meta['sha256'],encoder_mode=mode,
         dino_sha=meta.get('dino_sha256'),input_resolution=256,preprocessing_fingerprint=digest(preprocess),
         preprocessing=preprocess,query_file_list_hash=file_list_hash(queries),gallery_file_list_hash=file_list_hash(gallery),
         encoder_implementation_hash=digest({name:sha256(Path(__file__).with_name(name)) for name in
             ('eval_ocl_image_retrieval.py','dino_backbone.py','orbitformer.py','view_tokenizer.py')}),
         torch_version=str(torch.__version__))
    # JSON canonicalization prevents tuple/list differences from invalidating valid caches.
    return json.loads(json.dumps(identity, sort_keys=True))


def read_cache(folder,identity,queries,gallery):
    meta=folder/'cache_meta.json'
    if not meta.is_file():return None
    saved=json.loads(meta.read_text())
    if saved.get('identity')!=identity:
        print('Cache fingerprint changed; re-encoding (no stale reuse)',flush=True)
        return None
    outputs=[]
    for pool,entries in [('query',queries),('gallery',gallery)]:
        for suffix in ('descriptors.npy','labels.npy','paths.txt'):
            p=folder/f'{pool}_{suffix}'
            if not p.is_file() or sha256(p)!=saved['files'].get(p.name):
                raise ValueError(f'Corrupt/incomplete descriptor cache: {p}')
        values=np.load(folder/f'{pool}_descriptors.npy',allow_pickle=False)
        labels=np.load(folder/f'{pool}_labels.npy',allow_pickle=False).tolist()
        paths=(folder/f'{pool}_paths.txt').read_text().splitlines()
        if labels!=[label for _,label in entries] or paths!=[str(p) for p,_ in entries]:
            raise ValueError('Cache labels/paths differ from exact evaluation order')
        if values.shape!=(len(entries),768) or values.dtype!=np.float32 or not np.isfinite(values).all():
            raise ValueError('Invalid cached descriptor shape/dtype/values')
        if not np.allclose(np.linalg.norm(values,axis=1),1.,atol=1e-4):
            raise ValueError('Cache descriptors are not L2-normalized')
        outputs.append(torch.from_numpy(values))
    print(f'Descriptor cache hit: {folder}',flush=True)
    return tuple(outputs)


def write_cache(folder,identity,q,g,queries,gallery):
    folder.mkdir(parents=True,exist_ok=True)
    # Metadata is a commit marker; never expose a partially replaced set as complete.
    meta=folder/'cache_meta.json'
    if meta.exists():meta.unlink()
    files={}
    for pool,values,entries in [('query',q,queries),('gallery',g,gallery)]:
        for suffix,array in [('descriptors.npy',values.numpy()),('labels.npy',np.asarray([x[1] for x in entries]))]:
            target=folder/f'{pool}_{suffix}';temp=target.with_suffix(target.suffix+'.tmp')
            with temp.open('wb') as f:np.save(f,array,allow_pickle=False)
            temp.replace(target);files[target.name]=sha256(target)
        target=folder/f'{pool}_paths.txt';temp=target.with_suffix('.txt.tmp')
        temp.write_text('\n'.join(str(p) for p,_ in entries)+'\n');temp.replace(target)
        files[target.name]=sha256(target)
    dump(meta,dict(identity=identity,files=files))


def write_csv(path,rows,fields=None):
    path=guard(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


