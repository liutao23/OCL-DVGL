"""Render direct camera checks and KML-free canonical closure views."""
from __future__ import annotations
import json
import numpy as np
from PIL import Image
from common import guard, dump, fatal_cuda_error
from orbit_geometry import canonical_lookat, resolve_tilts
from gs_camera import Renderer

def _parse_degrees(text, name):
    try:
        values = [float(x.strip()) for x in str(text).split(',') if x.strip()]
    except ValueError as exc:
        raise ValueError(f'{name} must be comma-separated degrees') from exc
    if not values:
        raise ValueError(f'{name} cannot be empty')
    return values

def closure_spec(args, geometry):
    headings = _parse_degrees(args.closure_headings, '--closure-headings')
    tilts = resolve_tilts(args.closure_tilts, geometry)
    if any((not 0 <= x < 360 for x in headings)):
        raise ValueError('Closure headings must be in [0,360)')
    if any((not 0 <= x <= 90 for x in tilts)):
        raise ValueError('Closure tilts must be in [0,90]')
    return (headings, tilts)

def save_render(renderer, camera, path, center=None, rotation=None):
    rgb, diagnostics = renderer.render(camera, center, rotation)
    Image.fromarray(rgb).save(guard(path))
    stats = {}
    for key, array in diagnostics.items():
        np.save(guard(path.with_name(path.stem + '_' + key + '.npy')), array)
        finite = np.isfinite(array)
        stats[key] = dict(finite_fraction=float(finite.mean()), min=float(array[finite].min()) if finite.any() else None, max=float(array[finite].max()) if finite.any() else None)
        if key in ('alpha', 'accumulation') and finite.any():
            stats[key]['mean'] = float(array[finite].mean())
    return (rgb, stats)

def _closure_pose(geometry, heading, tilt):
    distance = float(geometry['median_slant_range'])
    C, Q = canonical_lookat(heading, tilt, distance, geometry)
    return (C, Q, distance)

def render_scene(args, assets, cameras, geometry, report):
    sid, out = (assets['scene_id'], args.output_dir)
    report['stage'] = 'model_load'
    renderer = Renderer(args.root, assets['model'], assets['ply'], args.device)
    report['renderer_settings'] = renderer.settings
    folder = guard(out / 'rerender' / sid)
    folder.mkdir(parents=True, exist_ok=True)
    n = len(cameras)
    selected = sorted(set((int(x) for x in np.linspace(0, n - 1, min(4, n)).round())))
    cells = []
    report['stage'] = 'direct_rerender'
    report['direct_gs_rerender'] = 'fail'
    matched = assets.get('matched_indices', list(range(n)))
    for j in selected:
        source_index = matched[j] if j < len(matched) else j
        with Image.open(guard(assets['images'][source_index])) as im:
            original = im.convert('RGB').copy()
        rgb, _ = save_render(renderer, cameras[j], folder / f'direct_{j:03d}.png')
        cells.extend([(original, f'{sid} original {source_index}: {assets['images'][source_index].name}'), (rgb, f'direct GS registered camera {j}')])
    report['direct_gs_rerender'] = 'pass_execution_only'
    report['visual_composition'] = 'pending_user_inspection'
    render_closure_views(args, assets, cameras, geometry, report, renderer=renderer)

def render_closure_views(args, assets, cameras, geometry, report, renderer=None):
    sid, out = (assets['scene_id'], args.output_dir)
    report['stage'] = 'closure'
    closure_dir = guard(out / 'closure' / sid)
    closure_dir.mkdir(parents=True, exist_ok=True)
    headings, tilts = closure_spec(args, geometry)
    camera = dict(cameras[0])
    for key in ('FoVx', 'FoVy'):
        camera[key] = float(np.median([c[key] for c in cameras]))
    for key in ('width', 'height'):
        camera[key] = int(np.median([c[key] for c in cameras]))
    previous = {}
    old_meta = closure_dir / 'metadata.json'
    if getattr(args, 'resume', False) and old_meta.is_file():
        previous = {int(r['view_id']): r for r in json.loads(old_meta.read_text()).get('renders', []) if 'view_id' in r}
    metadata, cells = ([], [])
    view_id = 0
    for tilt in tilts:
        for heading in headings:
            C, Q, distance = _closure_pose(geometry, heading, tilt)
            path = closure_dir / f'closure_{view_id:03d}.png'
            entry = dict(view_id=view_id, heading=float(heading), tilt=float(tilt), distance=float(distance), geometry_source=geometry.get('geometry_source', 'unknown'), camera_center=np.asarray(C).tolist(), camera_to_world=np.asarray(Q).tolist(), target=geometry.get('target'), FoV_degrees=[float(np.rad2deg(camera['FoVx'])), float(np.rad2deg(camera['FoVy']))], image_size=[camera['width'], camera['height']], render_success=False, file=str(path), source_policy='canonical orbit in the same COLMAP/3DGS similarity frame')
            old = previous.get(view_id)
            if old and old.get('render_success') and path.is_file() and path.stat().st_size:
                metadata.append(old)
                report['closure_produced'] += 1
                view_id += 1
                continue
            try:
                if renderer is None:
                    renderer = Renderer(args.root, assets['model'], assets['ply'], args.device)
                    report['renderer_settings'] = renderer.settings
                rgb, stats = save_render(renderer, camera, path, C, Q)
                entry.update(render_success=True, diagnostics=stats)
                report['closure_produced'] += 1
            except Exception as exc:
                rgb = None
                entry['error'] = f'{type(exc).__name__}: {exc}'
                report['closure_failed'] += 1
                if fatal_cuda_error(exc):
                    metadata.append(entry)
                    dump(closure_dir / 'metadata.json', dict(schema='ocl_plus_closure_v1', scene_id=sid, renders=metadata, geometry_source=geometry.get('geometry_source'), satellite_pixels_accessed=False, aborted_due_to_cuda_error=True))
                    raise
            metadata.append(entry)
            cells.append((rgb, f'h={heading:g} tilt={tilt:g} range={distance:.3f}' + (' FAILED' if rgb is None else '')))
            dump(closure_dir / 'metadata.json', dict(schema='ocl_plus_closure_v1', scene_id=sid, renders=metadata, geometry_source=geometry.get('geometry_source'), satellite_pixels_accessed=False))
            view_id += 1
    dump(closure_dir / 'metadata.json', dict(schema='ocl_plus_closure_v1', scene_id=sid, renders=metadata, geometry_source=geometry.get('geometry_source'), satellite_pixels_accessed=False))
    report['closure_count_expected'] = len(headings) * len(tilts)
    report['closure_headings'] = headings
    report['closure_tilts'] = tilts
    if getattr(args, 'full_build', False):
        if report['closure_failed']:
            raise RuntimeError(f'{report['closure_failed']} closure renders failed')
        return
    if cells:
        columns = max(1, len(headings))
    if report['closure_failed']:
        raise RuntimeError(f'{report['closure_failed']} closure renders failed; inspect metadata.json')
