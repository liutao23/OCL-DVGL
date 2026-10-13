"""OCL+ full source manifests with dataset-agnostic observation counts."""
from __future__ import annotations

import hashlib
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import guard, natural_key
from ocl_data import OrbitData, CURRICULUM
from dataset_layout import dataset_path



def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def development_split(ids,seed=0):
    ids=sorted(ids,key=natural_key)
    if len(ids)<4: raise ValueError('Need at least four complete source places for development')
    shuffled=np.random.default_rng(seed).permutation(ids).tolist()
    nval=70 if len(ids)==701 else min(len(ids)-2,max(2,int(math.ceil(.1*len(ids)))))
    return sorted(shuffled[nval:],key=natural_key),sorted(shuffled[:nval],key=natural_key)


def _counts_for_step(ids,rng,progress):
    policy=next(c for c in CURRICULUM if progress<c['until'])
    if progress>=.60:
        expected=.55*len(ids)
        singles=int(np.floor(expected))+int(rng.random()<expected%1)
        singles=max(int(np.ceil(.5*len(ids))),singles)
        counts=[1]*singles+rng.choice([4,2],len(ids)-singles,p=[1/3,2/3]).tolist()
        rng.shuffle(counts)
        return counts
    return rng.choice(policy['m'],len(ids),p=policy['p']).tolist()


def _sample_indices(rng,m,real_count,total_count,prefer_real=.7):
    if real_count<1 or total_count<real_count: raise ValueError('Invalid scene observation counts')
    pool_size=real_count if rng.random()<prefer_real else total_count
    m=min(int(m),pool_size)
    result=rng.choice(pool_size,m,replace=False).tolist()
    if m>1 and not any(i<real_count for i in result):
        result[0]=int(rng.integers(real_count))
    return result


def _closure_paths(folder):
    metadata=guard(folder/'metadata.json')
    if not metadata.is_file(): raise FileNotFoundError(metadata)
    payload=json.loads(metadata.read_text())
    rows=sorted(payload.get('renders',[]),key=lambda r:int(r['view_id']))
    if not rows or [int(r['view_id']) for r in rows]!=list(range(len(rows))):
        raise ValueError(f'{folder}: closure metadata has missing/duplicate view_id')
    paths=[]
    for row in rows:
        if not row.get('render_success'): raise ValueError(f'{folder}: closure view {row.get("view_id")} failed')
        p=guard(Path(row['file']))
        if not p.is_file() or p.stat().st_size<=0: raise ValueError(f'Missing closure image: {p}')
        paths.append(p)
    return paths


class FullOrbitData(OrbitData):
    def __init__(self,root,orbit_root,variant='mean_teacher',seed=0,final_refit=False,split_seed=None):
        self.root,self.smoke=guard(root),guard(orbit_root)
        self.report_path=guard(self.smoke/'orbit_manifest.json')
        report=json.loads(self.report_path.read_text())
        if report.get('schema') not in ('ocl_plus_orbit_v2','ocl_orbit_full_v1'):
            raise ValueError('Unsupported orbit manifest')
        self.paths,self.all_paths,self.real_count,self.closure_count={}, {}, {}, {}
        signatures=[]
        records=report['scenes']
        success=[r['scene_id'] for r in records if r['status']=='complete']
        if len(set(success))!=len(success) or set(success)!=set(report['successful_scene_ids']):
            raise ValueError('Inconsistent/duplicate successful IDs')
        records_by_id={r['scene_id']:r for r in records}
        for sid in sorted(success,key=natural_key):
            row=records_by_id[sid]
            if report.get('schema')=='ocl_plus_orbit_v2':
                folder=guard(Path(row['real_image_dir']))
            else:
                folder=guard(dataset_path(self.root)/'train/drone'/sid)
            real=sorted([guard(p) for p in folder.iterdir() if p.suffix.lower() in
                {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'} and guard(p).is_file()],key=lambda p:natural_key(p.name))
            closure=_closure_paths(self.smoke/'closure'/sid) if report.get('schema')=='ocl_plus_orbit_v2' else [
                guard(self.smoke/'closure'/sid/f'heading_{h:03d}_tilt_{t:03d}.png')
                for t in (35,25,15,5) for h in (0,90,180,270)]
            certified_real=int(row.get('real_image_count',len(real)))
            certified_closure=int(row.get('closure_count',len(closure)))
            if len(real)!=certified_real or len(closure)!=certified_closure or not real or not closure:
                raise ValueError(f'{sid}: manifest/assets disagree ({len(real)} real, {len(closure)} closure)')
            self.real_count[sid],self.closure_count[sid]=len(real),len(closure)
            self.all_paths[sid]=real+closure
            self.paths[sid]=real+closure
            for p in self.all_paths[sid]:
                st=p.stat();signatures.append((str(p),st.st_size,st.st_mtime_ns))
        self.ids=sorted(self.paths,key=natural_key)
        effective_split_seed=seed if split_seed is None else split_seed
        if final_refit:
            if len(self.ids)<2: raise ValueError('Refit needs at least two source places')
            self.train,self.val=self.ids[:],[]
        else:
            self.train,self.val=development_split(self.ids,effective_split_seed)
        self.variant,self.transform=variant,None
        self.orbit_hash=hashlib.sha256(self.report_path.read_bytes()).hexdigest()
        self.input_hash=digest(signatures)
        self.split=dict(seed=effective_split_seed,train=self.train,val=self.val,usable_count=len(self.ids),
                        train_count=len(self.train),val_count=len(self.val),final_refit=final_refit)
        self.split_hash=digest(self.split)

    def selections(self,ids,rng,progress):
        counts=_counts_for_step(ids,rng,progress)
        result=[]
        for sid,m in zip(ids,counts):
            indices=_sample_indices(rng,m,self.real_count[sid],len(self.paths[sid]))
            result.append((sid,indices))
        return result

    def validation_indices(self,sid,pool):
        real=self.real_count[sid];closure=self.closure_count[sid]
        if pool=='real': return list(range(real))
        if pool=='closure': return list(range(real,real+closure))
        raise ValueError(pool)
