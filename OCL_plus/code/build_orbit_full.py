#!/usr/bin/env python3
"""Build OCL+ closure assets for all scenes without requiring KML."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from common import guard, dump, natural_key
from prepare_orbit import discover, alignment, finalize_paths
from dataset_layout import add_dataset_arguments


@lru_cache(maxsize=None)
def _code_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_stamp(item, args):
    files=[item['ply'],*item['images']]
    if item.get('json'): files.append(item['json'])
    elif item.get('sparse'): files += [p for p in guard(item['sparse']).iterdir() if p.suffix in ('.bin','.txt')]
    if args.geometry_source=='kml' and item.get('kml') and item['kml'].is_file(): files.append(item['kml'])
    cfg=guard(item['model']/'cfg_args')
    if cfg.is_file(): files.append(cfg)
    code_names=['build_orbit_full.py','prepare_orbit.py','colmap_utils.py','orbit_geometry.py',
                'render_closure.py','gs_camera.py']
    code=[(name,_code_hash(Path(__file__).with_name(name))) for name in code_names]
    standalone=args.root/'3DGS'/'train_standalone.py'
    code.append(('standalone_renderer',_code_hash(standalone)))
    signatures=[(str(guard(p)),guard(p).stat().st_size,guard(p).stat().st_mtime_ns) for p in files if p and guard(p).is_file()]
    settings=dict(geometry_source=args.geometry_source,min_match_ratio=args.min_match_ratio,
                  min_matched_cameras=args.min_matched_cameras,min_real_images=args.min_real_images,
                  closure_headings=args.closure_headings,closure_tilts=args.closure_tilts)
    return hashlib.sha256(json.dumps([signatures,code,settings],sort_keys=True).encode()).hexdigest()


def closure_complete(out,sid):
    folder=guard(out/'closure'/sid)
    try:
        payload=json.loads((folder/'metadata.json').read_text())
        rows=payload['renders']
        ids=[int(r['view_id']) for r in rows]
        return bool(rows) and ids==list(range(len(rows))) and all(
            r.get('render_success') and guard(Path(r['file'])).is_file() and guard(Path(r['file'])).stat().st_size>0
            for r in rows)
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):
        return False


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(p)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--images-root',type=Path)
    p.add_argument('--gs-root',type=Path)
    p.add_argument('--scene-manifest',type=Path)
    p.add_argument('--tour-root',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('OCL_plus/outputs/orbit_full'))
    p.add_argument('--geometry-source',choices=('sfm',),default='sfm')
    p.add_argument('--min-match-ratio',type=float,default=.90)
    p.add_argument('--min-matched-cameras',type=int,default=8)
    p.add_argument('--min-real-images',type=int,default=8)
    p.add_argument('--closure-headings',default='0,90,180,270')
    p.add_argument('--closure-tilts',default='35,25,15,5',help='Comma-separated degrees or auto')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--scene-ids',help='Comma-separated scene IDs')
    p.add_argument('--worker-scene',help=argparse.SUPPRESS)
    a=finalize_paths(p.parse_args())
    if a.worker_scene:
        a.scene_filter=[a.worker_scene]
    a.full_build=True
    return a


def record(item,a):
    return dict(scene_id=item['scene_id'],status='failed',reason=item['audit']['notes'],
                geometry_source=a.geometry_source,real_image_count=item['audit']['real_image_count'],
                matched_camera_count=item['audit'].get('matched_camera_count',0),closure_count=0,
                real_image_dir=str(item.get('images_dir','')),gaussian_path=str(item.get('ply') or ''),
                orbit_metadata_path=str(a.output_dir/'alignment'/f"{item['scene_id']}_orbit.json"),
                kml_path=str(item.get('kml') or ''))


def worker(a):
    from render_closure import render_closure_views
    items=discover(a)
    if len(items)!=1:
        raise ValueError(f'Worker expected one scene, found {len(items)}')
    item=items[0];result=record(item,a)
    try:
        if not item['audit']['usable']: raise ValueError(item['audit']['notes'])
        result['fingerprint']=source_stamp(item,a)
        cameras,geometry=alignment(item,a.output_dir,result,a)
        result.update(closure_produced=0,closure_failed=0)
        render_closure_views(a,item,cameras,geometry,result)
        result.update(status='complete',reason='',closure_count=result['closure_produced'])
    except Exception as exc:
        result.update(reason=f'{type(exc).__name__}: {exc}',closure_count=result.get('closure_produced',0))
    dump(a.output_dir/'logs'/f'{a.worker_scene}.json',result)
    return 0 if result['status']=='complete' else 1


def _worker_command(a,sid):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(a.root),'--output-dir',str(a.output_dir),
         '--geometry-source',a.geometry_source,'--min-match-ratio',str(a.min_match_ratio),
         '--min-matched-cameras',str(a.min_matched_cameras),'--min-real-images',str(a.min_real_images),
         '--closure-headings',a.closure_headings,'--closure-tilts',a.closure_tilts,'--device',a.device,
         '--worker-scene',sid]
    if a.scene_manifest: cmd += ['--scene-manifest',str(a.scene_manifest)]
    else:
        cmd += ['--images-root',str(a.images_root),'--gs-root',str(a.gs_root)]
        if a.tour_root: cmd += ['--tour-root',str(a.tour_root)]
    if a.resume: cmd.append('--resume')
    return cmd


def main():
    a=arguments()
    for sub in ('logs','alignment','closure'):
        (a.output_dir/sub).mkdir(parents=True,exist_ok=True)
    if a.worker_scene: return worker(a)
    assets=discover(a)
    if a.scene_filter:
        allowed=set(a.scene_filter);assets=[x for x in assets if x['scene_id'] in allowed]
    assets=sorted(assets,key=lambda x:natural_key(x['scene_id']))
    results=[]
    def persist():
        dump(a.output_dir/'orbit_manifest.json',dict(
            schema='ocl_plus_orbit_v2',geometry_source=a.geometry_source,satellite_pixels_accessed=False,
            candidate_count=len(assets),min_camera_match_ratio=a.min_match_ratio,
            closure_spec=dict(headings=a.closure_headings,tilts=a.closure_tilts),
            successful_scene_ids=[r['scene_id'] for r in results if r['status']=='complete'],
            failed_scene_ids=[r['scene_id'] for r in results if r['status']!='complete'],scenes=results))
    for item in assets:
        sid=item['scene_id'];row=record(item,a);path=a.output_dir/'logs'/f'{sid}.json'
        if not item['audit']['usable']:
            results.append(row);persist();print(f'{sid}: failed {row["reason"]}',flush=True);continue
        stamp=source_stamp(item,a)
        if path.exists():
            old=json.loads(path.read_text())
            if not a.resume: raise FileExistsError(f'{path}: pass --resume')
            if old.get('fingerprint')!=stamp:
                row['reason']='Input/geometry fingerprint changed; use a new output directory'
                results.append(row);persist();continue
            if old.get('status')=='complete' and closure_complete(a.output_dir,sid):
                results.append(old);persist();print(f'{sid}: complete, skipped',flush=True);continue
        elif (a.output_dir/'closure'/sid).exists() and any((a.output_dir/'closure'/sid).glob('*.png')):
            row['reason']='Untracked existing closure: refusing mixed provenance; use a new output directory'
            results.append(row);persist();continue
        row.update(fingerprint=stamp,status='running',reason='');dump(path,row)
        with (a.output_dir/'logs'/f'{sid}.log').open('a') as log:
            done=subprocess.run(_worker_command(a,sid),stdout=log,stderr=subprocess.STDOUT)
        row=json.loads(path.read_text())
        if done.returncode or not closure_complete(a.output_dir,sid):
            row.update(status='failed',reason=row.get('reason') or f'Worker exit {done.returncode}; see {sid}.log')
            dump(path,row)
        results.append(row);persist();print(f"{sid}: {row['status']} {row.get('reason','')}",flush=True)
    persist()
    return 1 if any(r['status']!='complete' for r in results) else 0


if __name__=='__main__':
    raise SystemExit(main())
