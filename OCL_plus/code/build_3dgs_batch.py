#!/usr/bin/env python3
"""Batch raw UAV scenes -> COLMAP SfM -> 3DGS using train_standalone.py.

This is the KML-free front end for OCL+.  It intentionally treats each scene
(or each scene-height pair in SUES-200) as an independent reconstruction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from common import guard, dump, natural_key


def resolve(root,value):
    p=Path(value).expanduser()
    return guard(p if p.is_absolute() else root/p)


def entries(args):
    if args.scene_manifest:
        payload=json.loads(args.scene_manifest.read_text())
        rows=payload.get('scenes',payload) if isinstance(payload,dict) else payload
        result=[]
        for row in rows:
            if not row.get('scene_id') or not row.get('images_dir'):
                raise ValueError('Manifest rows need scene_id and images_dir')
            sid=str(row['scene_id'])
            image_dir=resolve(args.root,row['images_dir'])
            output=resolve(args.root,row['gs_dir']) if row.get('gs_dir') else guard(args.output_root/sid)
            result.append((sid,image_dir,output))
        return result
    if not args.images_root.is_dir(): raise FileNotFoundError(args.images_root)
    return [(p.name,guard(p),guard(args.output_root/p.name)) for p in
            sorted((x for x in args.images_root.iterdir() if x.is_dir()),key=lambda p:natural_key(p.name))]


def complete(output):
    cloud=output/'model'/'point_cloud'
    return cloud.is_dir() and any((p/'point_cloud.ply').is_file() and (p/'point_cloud.ply').stat().st_size>0
                                  for p in cloud.glob('iteration_*') if p.is_dir())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--images-root',type=Path,help='One subdirectory per scene')
    p.add_argument('--output-root',type=Path,default=Path('OCL_plus/outputs/3dgs'))
    p.add_argument('--scene-manifest',type=Path,help='Same manifest accepted by build_orbit_full.py')
    p.add_argument('--scene-ids',help='Comma-separated subset')
    p.add_argument('--iterations',type=int,default=30000)
    p.add_argument('--camera-model',default='SIMPLE_PINHOLE')
    p.add_argument('--colmap-executable',default='colmap')
    p.add_argument('--reuse-colmap',action='store_true')
    p.add_argument('--skip-existing',action='store_true')
    p.add_argument('--extra-args',nargs=argparse.REMAINDER,default=[],help='Arguments forwarded to train_standalone.py')
    a=p.parse_args();a.root=guard(a.root)
    a.images_root=resolve(a.root,a.images_root) if a.images_root else None
    a.output_root=resolve(a.root,a.output_root)
    a.scene_manifest=resolve(a.root,a.scene_manifest) if a.scene_manifest else None
    if bool(a.scene_manifest)==bool(a.images_root): p.error('Provide exactly one of --images-root or --scene-manifest')
    if a.iterations<1: p.error('--iterations must be positive')
    wanted=set(s.strip() for s in a.scene_ids.split(',') if s.strip()) if a.scene_ids else None
    rows=[]
    standalone=guard(a.root/'3DGS重建'/'train_standalone.py')
    for sid,image_dir,output in entries(a):
        if wanted and sid not in wanted: continue
        row=dict(scene_id=sid,images_dir=str(image_dir),gs_dir=str(output),status='running')
        try:
            if complete(output):
                if a.skip_existing:
                    row['status']='complete_existing';rows.append(row);print(f'{sid}: existing',flush=True);continue
                raise FileExistsError(f'{output}: completed 3DGS exists; pass --skip-existing')
            cmd=[sys.executable,str(standalone),str(image_dir),'--output',str(output),
                 '--iterations',str(a.iterations),'--camera-model',a.camera_model,
                 '--colmap-executable',a.colmap_executable,'--disable_viewer']
            if a.reuse_colmap: cmd.append('--reuse-colmap')
            cmd += list(a.extra_args)
            output.parent.mkdir(parents=True,exist_ok=True)
            log=output.parent/f'{sid}_3dgs_batch.log'
            with log.open('a') as f:
                done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            if done.returncode or not complete(output):
                raise RuntimeError(f'3DGS worker failed ({done.returncode}); see {log}')
            row['status']='complete';row['log']=str(log)
        except Exception as exc:
            row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        rows.append(row);print(f"{sid}: {row['status']} {row.get('reason','')}",flush=True)
        dump(a.output_root/'batch_manifest.json',dict(schema='ocl_plus_3dgs_batch_v1',scenes=rows))
    dump(a.output_root/'batch_manifest.json',dict(schema='ocl_plus_3dgs_batch_v1',scenes=rows))
    return 1 if any(r['status']=='failed' for r in rows) else 0


if __name__=='__main__': raise SystemExit(main())
