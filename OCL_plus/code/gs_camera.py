"""Lazy import of the existing standalone Gaussian loader/renderer; never train."""
import importlib.util
import sys
from types import SimpleNamespace
import numpy as np
from common import guard, config


class Renderer:
    def __init__(self, root, model, ply, device):
        import torch
        self.torch = torch
        self.device = torch.device("cuda:0" if device == "cuda" else device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Existing Gaussian renderer requires CUDA; use --device cuda:0')
        torch.cuda.set_device(self.device)
        name = '_ocl_standalone_renderer'
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, guard(root/'3DGS'/'train_standalone.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(name, None)
                raise
        self.module = sys.modules[name]
        # Match training: accelerated rasterizers require a separate DC tensor.
        # Combined SH with empty DC can trigger an illegal CUDA memory access.
        self.separate_sh = bool(self.module.SPARSE_ADAM_AVAILABLE)
        cfg = config(model/'cfg_args')
        # Infer degree from the PLY header if cfg_args was not retained.
        with guard(ply).open('rb') as f:
            rest = 0
            for _ in range(10000):
                line = f.readline()
                if not line:
                    raise ValueError('Truncated PLY header')
                if line.startswith(b'property ') and b' f_rest_' in line:
                    rest += 1
                if line.strip() == b'end_header':
                    break
            else:
                raise ValueError('Invalid PLY header')
        degree = int(round(np.sqrt(rest/3+1)-1))
        if rest != 3*(degree+1)**2-3 or degree > 3:
            raise ValueError('Unsupported Gaussian SH degree')
        self.model = self.module.GaussianModel(degree)
        self.model.load_ply(str(guard(ply)), use_train_test_exp=False)
        if not torch.isfinite(self.model.get_xyz).all() or not len(self.model.get_xyz):
            raise ValueError('Empty/non-finite Gaussian model')
        self.pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                                    debug=False, antialiasing=cfg.get('antialiasing', False))
        self.background = torch.tensor([1., 1., 1.] if cfg.get('white_background', False) else [0., 0., 0.], device=self.device)
        self.settings = dict(separate_sh=self.separate_sh, sh_degree=degree, white_background=cfg.get('white_background', False),
                             antialiasing=self.pipe.antialiasing, exposure_applied=False,
                             note='cfg_args lacks pipeline flags: antialiasing defaults to training default False')

    def render(self, intrinsics, center=None, rotation=None):
        torch = self.torch
        c = intrinsics
        C = np.asarray(c['position'] if center is None else center)
        Q = np.asarray(c['rotation'] if rotation is None else rotation)
        if not np.isfinite(C).all() or not np.allclose(Q.T@Q, np.eye(3), atol=1e-5) or np.linalg.det(Q) <= 0:
            raise ValueError('Invalid camera pose')
        # Q is camera-to-world. COLMAP W2C=[Q.T | -Q.T C].
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = Q.T, -Q.T@C
        view = torch.tensor(w2c, dtype=torch.float32, device=self.device).T.contiguous()
        proj = self.module.getProjectionMatrix(znear=0.01, zfar=100., fovX=c['FoVx'], fovY=c['FoVy']).T.to(self.device)
        cam = SimpleNamespace(image_width=c['width'], image_height=c['height'], FoVx=c['FoVx'], FoVy=c['FoVy'],
                              world_view_transform=view, full_proj_transform=view@proj,
                              camera_center=torch.tensor(C, dtype=torch.float32, device=self.device), image_name=c['img_name'])
        with torch.no_grad():
            package = self.module.render(cam, self.model, self.pipe, self.background,
                                         separate_sh=self.separate_sh)
        rgb = package['render'].detach().cpu().numpy().transpose(1, 2, 0)
        if rgb.shape != (c['height'], c['width'], 3) or not np.isfinite(rgb).all():
            raise ValueError('Renderer returned invalid RGB')
        diagnostics = {}
        for key in ('depth', 'alpha', 'accumulation'):
            if key in package and package[key] is not None:
                diagnostics[key] = package[key].detach().cpu().numpy()
        return np.uint8(np.clip(rgb, 0, 1)*255+0.5), diagnostics
