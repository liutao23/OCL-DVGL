"""OCL+ geometry preparation.

Default path is KML-free: reconstructed COLMAP/3DGS cameras -> canonical orbit.
KML remains an optional comparison backend for University-1652 style metadata.
No satellite pixels are read by this module.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from common import guard, dump, config, fatal_cuda_error, natural_key
from colmap_utils import photos, read_cameras
from orbit_geometry import estimate_orbit
from dataset_layout import dataset_path, add_dataset_arguments

def _resolve(root, value):
    p = Path(value).expanduser()
    return guard(p if p.is_absolute() else root / p)

def _manifest_entries(args):
    if getattr(args, 'scene_manifest', None):
        payload = json.loads(guard(args.scene_manifest).read_text())
        rows = payload.get('scenes', payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError('Scene manifest must be a list or {"scenes": [...]}')
        entries = []
        for row in rows:
            if not isinstance(row, dict) or not row.get('scene_id') or (not row.get('images_dir')) or (not row.get('gs_dir')):
                raise ValueError('Each scene manifest row needs scene_id, images_dir and gs_dir')
            entries.append(dict(scene_id=str(row['scene_id']), images_dir=_resolve(args.root, row['images_dir']), gs_dir=_resolve(args.root, row['gs_dir']), kml=_resolve(args.root, row['kml']) if row.get('kml') else None))
        return entries
    if not args.images_root or not args.gs_root:
        raise ValueError('Provide --images-root and --gs-root, or --scene-manifest')
    entries = []
    if not args.images_root.is_dir():
        raise FileNotFoundError(args.images_root)
    for image_dir in sorted((guard(p) for p in args.images_root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        sid = image_dir.name
        entries.append(dict(scene_id=sid, images_dir=image_dir, gs_dir=guard(args.gs_root / sid), kml=guard(args.tour_root / f'Tour_{sid}.kml') if getattr(args, 'tour_root', None) else None))
    return entries

def _locate_model(directory):
    model_options = [guard(directory / 'model'), guard(directory)]
    return next((m for m in model_options if guard(m / 'point_cloud').is_dir()), model_options[0])

def _locate_ply(model):
    cloud = guard(model / 'point_cloud')
    candidates = []
    if cloud.is_dir():
        for iteration in cloud.iterdir():
            iteration = guard(iteration)
            if iteration.is_dir() and iteration.name.startswith('iteration_'):
                try:
                    number = int(iteration.name.split('_')[-1])
                except ValueError:
                    continue
                ply = guard(iteration / 'point_cloud.ply')
                if ply.is_file() and ply.stat().st_size > 0:
                    candidates.append((number, ply))
    return max(candidates)[1] if candidates else None

def _locate_sparse(directory, model):
    options = [directory / 'workspace' / 'sparse' / '0', directory / 'workspace' / 'sparse', directory / 'dataset' / 'sparse' / '0', directory / 'dataset' / 'sparse', directory / 'sparse' / '0', model / 'sparse' / '0']
    cfg = config(model / 'cfg_args')
    if cfg.get('source_path'):
        source = guard(cfg['source_path'])
        options += [source / 'sparse' / '0', source / 'sparse']
    for sparse in options:
        sparse = guard(sparse)
        if any((all((guard(sparse / (name + ext)).is_file() for name in ('cameras', 'images'))) for ext in ('.bin', '.txt'))):
            return sparse
    return None

def discover(args):
    rows, assets = ([], [])
    entries = _manifest_entries(args)
    scene_filter = set(getattr(args, 'scene_filter', None) or [])
    seen = set()
    for entry in entries:
        sid = entry['scene_id']
        if scene_filter and sid not in scene_filter:
            continue
        if sid in seen:
            raise ValueError(f'Duplicate scene_id: {sid}')
        seen.add(sid)
        row = dict(scene_id=sid, real_image_count=0, matched_camera_count=0, kml_present=False, cameras_json_present=False, gs_ply_present=False, colmap_sparse_present=False, usable=False, notes='')
        item = dict(scene_id=sid, directory=entry['gs_dir'], images_dir=entry['images_dir'], kml=entry.get('kml'))
        try:
            item['images'] = photos(entry['images_dir'])
            row['real_image_count'] = len(item['images'])
            if len(item['images']) < getattr(args, 'min_real_images', 8):
                raise ValueError(f'Need at least {args.min_real_images} real images, found {len(item['images'])}')
            item['kml'] = guard(item['kml']) if item.get('kml') else None
            row['kml_present'] = bool(item['kml'] and item['kml'].is_file())
            item['model'] = _locate_model(entry['gs_dir'])
            camera_json = guard(item['model'] / 'cameras.json')
            item['json'] = camera_json if camera_json.is_file() else None
            row['cameras_json_present'] = item['json'] is not None
            item['ply'] = _locate_ply(item['model'])
            row['gs_ply_present'] = item['ply'] is not None
            item['sparse'] = _locate_sparse(entry['gs_dir'], item['model'])
            row['colmap_sparse_present'] = item['sparse'] is not None
            if item['ply'] is None:
                raise ValueError('Missing trained Gaussian point_cloud.ply')
            if item['json'] is None and item['sparse'] is None:
                raise ValueError('Missing cameras.json and COLMAP sparse cameras')
            if getattr(args, 'geometry_source', 'sfm') == 'kml' and (not row['kml_present']):
                raise ValueError('geometry-source=kml requires a KML path')
            matched, indices = read_cameras(item['json'], item['sparse'], item['images'], min_match_ratio=getattr(args, 'min_match_ratio', 0.9), return_indices=True, min_count=getattr(args, 'min_matched_cameras', 8))
            item['matched_indices'] = indices
            row['matched_camera_count'] = len(matched)
            row['usable'] = True
            row['notes'] = 'KML-free capable; visual Gaussian/closure validity still requires inspection'
        except Exception as exc:
            row['notes'] = f'{type(exc).__name__}: {exc}'
        item['audit'] = row
        rows.append(row)
        assets.append(item)
    if getattr(args, 'output_dir', None) and (not getattr(args, 'full_build', False)):
        path = guard(args.output_dir / 'audit' / 'scenes.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['scene_id', 'real_image_count', 'matched_camera_count', 'usable', 'notes'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    return assets

def alignment(item, out, report, args):
    """Return matched GS cameras and a renderable canonical geometry object."""
    sid = item['scene_id']
    cameras, indices = read_cameras(item['json'], item['sparse'], item['images'], min_match_ratio=args.min_match_ratio, return_indices=True, min_count=args.min_matched_cameras)
    centers = np.stack([c['position'] for c in cameras])
    rotations = np.stack([c['rotation'] for c in cameras])
    np.savez(guard(out / 'alignment' / f'{sid}_gs_cameras.npz'), centers=centers, camera_to_world=rotations, matched_indices=indices, matched_filenames=[item['images'][i].name for i in indices], **{k: np.array([c[k] for c in cameras]) for k in ('fx', 'fy', 'FoVx', 'FoVy', 'width', 'height')})
    geometry = estimate_orbit(cameras)
    geometry.update(geometry_source='sfm', matched_indices=indices, matched_filenames=[item['images'][i].name for i in indices])
    dump(out / 'alignment' / f'{sid}_orbit.json', geometry)
    report.update(matched_camera_count=len(cameras), geometry_source='sfm', geometry_diagnostics={k: geometry[k] for k in ('median_slant_range', 'slant_range_q10_q50_q90', 'observed_tilt_q10_q50_q90', 'ray_residual_q50_q90_max', 'forward_to_target_cos_q10_q50')})
    return (cameras, geometry)

def fingerprint(item, args):
    files = [item['ply'], *item['images'], *sorted(Path(__file__).parent.glob('*.py')), args.root / '3DGS' / 'train_standalone.py']
    if item.get('kml') and item['kml'].is_file() and (args.geometry_source == 'kml'):
        files.append(item['kml'])
    files += [item['json']] if item['json'] else [p for p in guard(item['sparse']).iterdir() if p.suffix in ('.bin', '.txt')]
    cfg = guard(item['model'] / 'cfg_args')
    if cfg.is_file():
        files.append(cfg)
    signatures = [(str(guard(p)), guard(p).stat().st_size, guard(p).stat().st_mtime_ns) for p in files if p and guard(p).is_file()]
    settings = [args.geometry_source, args.min_match_ratio, args.min_matched_cameras, getattr(args, 'closure_headings', None), getattr(args, 'closure_tilts', None)]
    return hashlib.sha256(json.dumps([signatures, settings], sort_keys=True, default=str).encode()).hexdigest()

def _base_parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(p)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--images-root', type=Path, help='Directory containing one subdirectory per scene')
    p.add_argument('--gs-root', type=Path, help='Directory containing one 3DGS output subdirectory per scene')
    p.add_argument('--scene-manifest', type=Path, help='JSON list of {scene_id, images_dir, gs_dir[, kml]}')
    p.add_argument('--tour-root', type=Path, help='Optional KML root; only used with --geometry-source kml')
    p.add_argument('--output-dir', type=Path, default=Path('OCL_plus/outputs/smoke'))
    p.add_argument('--geometry-source', choices=('sfm',), default='sfm')
    p.add_argument('--min-match-ratio', type=float, default=0.9)
    p.add_argument('--min-matched-cameras', type=int, default=8)
    p.add_argument('--min-real-images', type=int, default=8)
    p.add_argument('--scene-ids', help='Comma-separated scene IDs')
    p.add_argument('--num-scenes', type=int, default=12)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--closure-headings', default='0,90,180,270')
    p.add_argument('--closure-tilts', default='35,25,15,5', help='Comma-separated degrees or auto')
    p.add_argument('--resume', action='store_true')
    return p

def finalize_paths(args):
    args.root = guard(args.root)
    if args.scene_manifest:
        args.scene_manifest = _resolve(args.root, args.scene_manifest)
    else:
        dataset = dataset_path(args.root, getattr(args, 'dataset', 'U1652'), getattr(args, 'datasets_root', None))
        args.images_root = _resolve(args.root, args.images_root or dataset / 'train/drone')
        gs_name = 'University_3DGS' if dataset.name == 'U1652' else '3DGS'
        args.gs_root = _resolve(args.root, args.gs_root or dataset / 'train' / gs_name)
    args.tour_root = _resolve(args.root, args.tour_root) if args.tour_root else None
    args.output_dir = _resolve(args.root, args.output_dir)
    if not 0 < args.min_match_ratio <= 1 or args.min_matched_cameras < 3 or args.min_real_images < 3:
        raise ValueError('Invalid camera thresholds')
    args.scene_filter = [s.strip() for s in args.scene_ids.split(',') if s.strip()] if args.scene_ids else None
    return args

def main():
    p = _base_parser()
    p.add_argument('--audit-only', action='store_true')
    args = finalize_paths(p.parse_args())
    for sub in ('audit', 'alignment', 'rerender', 'closure', 'contact_sheets', 'logs'):
        guard(args.output_dir / sub).mkdir(parents=True, exist_ok=True)
    assets = discover(args)
    if args.audit_only:
        print(json.dumps([a['audit'] for a in assets], indent=2, ensure_ascii=False))
        return 0
    selected = [a for a in assets if a['audit']['usable']][:args.num_scenes]
    if args.scene_filter:
        by_id = {a['scene_id']: a for a in assets}
        selected = [by_id[s] for s in args.scene_filter if s in by_id]
    results = []
    for item in selected:
        sid = item['scene_id']
        row = dict(scene_id=sid, status='running', closure_produced=0, closure_failed=0, satellite_pixels_accessed=False, usable_assets=item['audit'])
        try:
            cameras, geometry = alignment(item, args.output_dir, row, args)
            from render_closure import render_scene
            render_scene(args, item, cameras, geometry, row)
            row['status'] = 'complete'
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}', fatal_cuda_error=fatal_cuda_error(exc))
            guard(args.output_dir / 'logs' / f'{sid}.traceback.txt').write_text(traceback.format_exc())
        dump(args.output_dir / 'logs' / f'{sid}.json', row)
        results.append(row)
    dump(args.output_dir / 'smoke_report.json', dict(schema='ocl_plus_smoke_v1', geometry_source=args.geometry_source, satellite_pixels_accessed=False, scenes=results))
    return 1 if any((r['status'] != 'complete' for r in results)) else 0
if __name__ == '__main__':
    sys.exit(main())
