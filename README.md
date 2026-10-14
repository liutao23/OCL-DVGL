# OCL
<img width="4410" height="2477" alt="图片1" src="https://github.com/user-attachments/assets/40a08828-a24e-4415-9dc3-8841aa15771d" />

> **Note:** The paper has undergone several revisions, and this repository corresponds to the final version.

## Directory Structure

```text
OCL_core_code/
├── OCL_plus/
│   ├── code/       # Data preparation, training, and evaluation code
│   ├── models/     # Pretrained weights (download separately)
│   └── outputs/    # Reconstructions, checkpoints, and evaluation results generated at runtime
├── 3DGS/
│   └── train_standalone.py
└── README.md
```

## Script Functions

| Script | Function |
|---|---|
| `build_3dgs_batch.py` | Batch COLMAP pose estimation and 3DGS reconstruction |
| `build_orbit_full.py` | Generate canonical 3DGS views and training data manifests |
| `train_ocl_full.py` | Train the OCL model and select checkpoints based on source-domain validation results |
| `eval_ocl_image_retrieval.py` | Evaluate image retrieval and report R@1, R@5, R@10, MRR, and mAP |
| `eval_ocl_weather.py` | Multi-weather Drone → Satellite evaluation on University-1652 |
| `iaa_trans.py` | Weather augmentation including fog, rain, snow, and illumination changes |
| `dino_backbone.py` | DINOv3 feature extraction and image preprocessing |
| `view_tokenizer.py`, `orbitformer.py` | View encoding and multi-view aggregation |
| `full_data.py`, `full_memory.py` | Data sampling and teacher feature memory maintenance |
| `losses.py` | Distillation loss, NCE loss, and EMA update |

Except for `3DGS/train_standalone.py`, all scripts above are located in `OCL_plus/code/`.

## Environment Setup

```bash
conda create -n ocl python=3.10 -y
conda activate ocl
pip install torch torchvision timm safetensors "numpy<2" pillow tqdm scipy "opencv-python<4.12" plyfile imgaug==0.4.0
```

An NVIDIA GPU is required. Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to select a CUDA version compatible with your driver. Reconstruction and rendering also require [COLMAP](https://colmap.github.io/install.html), together with the `simple-knn` and `diff-gaussian-rasterization` CUDA extensions from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).

DINOv3 ViT-B/16 pretrained weights can be downloaded from
[Hugging Face](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).

After downloading, place the weight file at:

```text
OCL_plus/models/DINOv3_ViTB16_LVD1689M.safetensors
```

## Datasets

- [University-1652 official download](https://github.com/zhxry/University1652)
- [SUES-200 official download](https://github.com/Reza-Zhu/SUES-200-Benchmark)
- [IRVL328 official download](https://github.com/liutao23/ODGNNLoc)

Download University-1652 and SUES-200 from the official repositories above, and obtain IRVL328 from its official release. The training and evaluation commands use the following data directory:

```text
../Datasets/U1652/
├── train/drone/<scene_id>/<image>
└── test/
    ├── query_drone/<scene_id>/<image>
    └── gallery_satellite/<scene_id>/<image>
```

## Training

First reconstruct 3DGS scenes and then generate canonical views. These two steps can be skipped if the corresponding outputs already exist.

```bash
python OCL_plus/code/build_3dgs_batch.py \
  --images-root ../Datasets/U1652/train/drone \
  --output-root OCL_plus/outputs/3dgs_u1652 \
  --iterations 7000 --skip-existing

python OCL_plus/code/build_orbit_full.py \
  --dataset U1652 --datasets-root ../Datasets \
  --gs-root OCL_plus/outputs/3dgs_u1652 \
  --output-dir OCL_plus/outputs/orbit_u1652 \
  --device cuda:0 --resume

python OCL_plus/code/train_ocl_full.py \
  --orbit-root OCL_plus/outputs/orbit_u1652 \
  --checkpoint OCL_plus/models/DINOv3_ViTB16_LVD1689M.safetensors \
  --seed 0 --split-seed 0 \
  --epochs 20 --steps-per-epoch 100 --batch-places 8 --place-microbatch 2 \
  --device cuda:0 --output-dir OCL_plus/outputs/train_full_ocl
```

## Evaluation

Drone → Satellite:

```bash
python OCL_plus/code/eval_ocl_image_retrieval.py \
  --checkpoint OCL_plus/outputs/train_full_ocl/checkpoints/best_image.pt \
  --encoder-mode ocl_image_model --expected-variant mean_teacher \
  --query-dir ../Datasets/U1652/test/query_drone \
  --gallery-dir ../Datasets/U1652/test/gallery_satellite \
  --output-dir OCL_plus/outputs/eval_full_ocl/drone_to_satellite \
  --device cuda:0 --batch-size 8
```

Metrics are saved to `OCL_plus/outputs/eval_full_ocl/drone_to_satellite/metrics.json`.

### Multi-Weather Evaluation

We evaluate nine weather conditions on University-1652: `fog`, `rain`, `snow`, `dark`, `light`, `fog_rain`, `fog_snow`, `rain_snow`, and `wind`. Only UAV query images are augmented, while the satellite gallery remains unchanged. Augmentation is performed online, and no weather-transformed images are saved.

```bash
python OCL_plus/code/eval_ocl_weather.py \
  --checkpoint OCL_plus/outputs/train_full_ocl/checkpoints/best_image.pt \
  --expected-variant mean_teacher \
  --dataset-root ../Datasets/U1652 \
  --output-dir OCL_plus/outputs/weather_full_ocl \
  --device cuda:0 --batch-size 8 --workers 8
```

To evaluate specific weather conditions, add `--weather fog rain snow`. To skip completed conditions, add `--resume`.

The summary results are saved to `OCL_plus/outputs/weather_full_ocl/weather_summary.csv`, and the metrics for each weather condition are saved to `drone_to_satellite/<weather>/metrics.json`.
<img width="2492" height="1839" alt="qualitative_analysis" src="https://github.com/user-attachments/assets/bc7cabbc-960c-4b55-adc9-5bc69a801cba" />

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{liu2026satellitefreetrainingdroneviewgeolocalization,
      title={Satellite-Free Training for Drone-View Geo-Localization}, 
      author={Tao Liu and Yingzhi Zhang and Xingrun Wang and Kan Ren and Xiaoqi Zhao},
      year={2026},
      eprint={2604.01581},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.01581}, 
}
```
