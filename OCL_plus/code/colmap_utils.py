"""Read GS JSON (camera-to-world) or undistorted COLMAP (world-to-camera)."""
import json
import struct
from pathlib import Path
import numpy as np
from common import guard, natural_key


def photos(folder):
    p = guard(folder)
    return sorted((guard(f) for f in p.iterdir() if guard(f).is_file() and f.suffix.lower() in
                   {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}), key=lambda f: natural_key(f.name)) if p.is_dir() else []


def validate(c):
    c = dict(c)
    c['position'] = np.asarray(c['position'], dtype=float)
    c['rotation'] = np.asarray(c['rotation'], dtype=float)
    Q = c['rotation']
    if c['position'].shape != (3,) or Q.shape != (3, 3) or not np.isfinite(c['position']).all():
        raise ValueError('Invalid camera center/rotation')
    if not np.allclose(Q.T @ Q, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(Q), 1, atol=1e-5):
        raise ValueError('Camera rotation is not a proper orthonormal camera-to-world matrix')
    for key in ('fx', 'fy', 'width', 'height'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'Invalid camera {key}')
    c['width'], c['height'] = int(c['width']), int(c['height'])
    c['FoVx'] = float(2*np.arctan(c['width']/(2*c['fx'])))
    c['FoVy'] = float(2*np.arctan(c['height']/(2*c['fy'])))
    return c


def quaternion(q):
    q = np.asarray(q, float)
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise ValueError('Non-unit COLMAP quaternion')
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def sparse_records(folder):
    p = guard(folder)
    intrinsics, images = {}, []
    def camera(cid, model, w, h, params):
        if model == 'SIMPLE_PINHOLE':
            fx, cx, cy = params
            fy = fx
        elif model == 'PINHOLE':
            fx, fy, cx, cy = params
        else:
            raise ValueError(f'COLMAP fallback requires undistorted PINHOLE; got {model}')
        # Match train_standalone's centered FoV renderer, which does not use cx/cy.
        intrinsics[cid] = dict(fx=fx, fy=fy, width=w, height=h,
                               colmap_cx=cx, colmap_cy=cy, principal_point_policy='same centered projection as training')
    if guard(p/'cameras.bin').is_file() and guard(p/'images.bin').is_file():
        def read(f, fmt):
            size = struct.calcsize('<'+fmt)
            data = f.read(size)
            if len(data) != size:
                raise ValueError('Truncated COLMAP binary')
            return struct.unpack('<'+fmt, data)
        with guard(p/'cameras.bin').open('rb') as f:
            for _ in range(read(f, 'Q')[0]):
                cid, mid, w, h = read(f, 'iiQQ')
                if mid not in (0, 1):
                    raise ValueError('Use undistorted COLMAP sparse model (PINHOLE only)')
                camera(cid, ('SIMPLE_PINHOLE', 'PINHOLE')[mid], w, h, read(f, 'd'*(3 if mid == 0 else 4)))
        with guard(p/'images.bin').open('rb') as f:
            for _ in range(read(f, 'Q')[0]):
                entry = read(f, 'idddddddi')
                name = bytearray()
                while True:
                    b = f.read(1)
                    if not b:
                        raise ValueError('Truncated COLMAP image name')
                    if b == b'\0':
                        break
                    name.extend(b)
                images.append((name.decode(), entry[1:5], entry[5:8], entry[8]))
                f.seek(read(f, 'Q')[0]*24, 1)
    else:
        with guard(p/'cameras.txt').open() as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                a = line.split()
                camera(int(a[0]), a[1], int(a[2]), int(a[3]), list(map(float, a[4:])))
        with guard(p/'images.txt').open() as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                a = line.split(maxsplit=9)
                images.append((a[9].strip(), list(map(float, a[1:5])), list(map(float, a[5:8])), int(a[8])))
                f.readline()  # one POINTS2D line, including an empty one
    records = []
    for name, q, t, cid in images:
        R = quaternion(q)
        records.append(dict(img_name=name, position=-R.T@np.asarray(t), rotation=R.T, **intrinsics[cid]))
    return records


def read_cameras(json_path, sparse, image_paths, min_match_ratio=1.0, return_indices=False, min_count=3):
    records = json.loads(guard(json_path).read_text()) if json_path else sparse_records(sparse)
    result = []
    used, indices = set(), []
    for index, image in enumerate(image_paths):
        candidates = [i for i, c in enumerate(records) if Path(c['img_name']).name == image.name]
        if not candidates:
            candidates = [i for i, c in enumerate(records) if Path(c['img_name']).stem == image.stem]
        if not candidates and min_match_ratio < 1.0:
            continue
        if len(candidates) != 1 or candidates[0] in used:
            raise ValueError(f'Camera name match missing/ambiguous: {image.name}')
        used.add(candidates[0])
        result.append(validate(records[candidates[0]]))
        indices.append(index)
    if len(image_paths) < min_count:
        raise ValueError(f'Need at least {min_count} source images, found {len(image_paths)}')
    if len(result) < min_count or len(result)/len(image_paths) < min_match_ratio:
        raise ValueError(f'Camera match ratio {len(result)}/{len(image_paths)} below {min_match_ratio} or fewer than {min_count} matches')
    return (result, indices) if return_indices else result
