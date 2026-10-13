# OCL

## 目录结构

```text
OCL_core_code/
├── OCL_plus/
│   ├── code/       # 数据准备、训练与评测代码
│   ├── models/     # 预训练权重（自行下载）
│   └── outputs/    # 运行时生成的重建、权重和评测结果
├── 3DGS重建/
│   └── train_standalone.py
└── README.md
```

## 脚本功能

| 脚本 | 功能 |
|---|---|
| `build_3dgs_batch.py` | 批量执行 COLMAP 位姿估计与 3DGS 重建 |
| `build_orbit_full.py` | 生成 canonical 3DGS views 和训练数据清单 |
| `train_ocl_full.py` | 训练 OCL 模型并根据源域验证结果选择权重 |
| `eval_ocl_image_retrieval.py` | 评测图像检索，输出 R@1、R@5、R@10、MRR 和 mAP |
| `eval_ocl_weather.py` | University-1652 多天气 Drone → Satellite 评测 |
| `iaa_trans.py` | 雾、雨、雪、光照等天气增强 |
| `dino_backbone.py` | DINOv3 特征提取与图像预处理 |
| `view_tokenizer.py`、`orbitformer.py` | 视图编码与多视图聚合 |
| `full_data.py`、`full_memory.py` | 数据采样与教师特征记忆维护 |
| `losses.py` | 蒸馏、NCE 损失与 EMA 更新 |

除 `3DGS重建/train_standalone.py` 外，上述脚本均位于 `OCL_plus/code/`。

## 环境准备

```bash
conda create -n ocl python=3.10 -y
conda activate ocl
pip install torch torchvision timm safetensors "numpy<2" pillow tqdm scipy "opencv-python<4.12" plyfile imgaug==0.4.0
```

需要 NVIDIA GPU。按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)选择匹配驱动的 CUDA 版本。重建和渲染还需安装 [COLMAP](https://colmap.github.io/install.html)，以及 [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 中的 `simple-knn` 和 `diff-gaussian-rasterization` CUDA 扩展。

DINOv3 ViT-B/16 预训练权重路径：

```text
OCL_plus/models/DINOv3_ViTB16_LVD1689M.safetensors
```

## 数据集

- [University-1652 官方下载入口](https://github.com/zhxry/University1652)
- [SUES-200 官方下载入口](https://github.com/Reza-Zhu/SUES-200-Benchmark)

数据集下载入口见上述官方仓库。训练与测试命令使用以下数据目录：

```text
../Datasets/U1652/
├── train/drone/<场景编号>/<图像>
└── test/
    ├── query_drone/<场景编号>/<图像>
    └── gallery_satellite/<场景编号>/<图像>
```

## 训练

先重建 3DGS，再生成 canonical views；已有对应产物时可跳过这两步。

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

## 测试

Drone → Satellite：

```bash
python OCL_plus/code/eval_ocl_image_retrieval.py \
  --checkpoint OCL_plus/outputs/train_full_ocl/checkpoints/best_image.pt \
  --encoder-mode ocl_image_model --expected-variant mean_teacher \
  --query-dir ../Datasets/U1652/test/query_drone \
  --gallery-dir ../Datasets/U1652/test/gallery_satellite \
  --output-dir OCL_plus/outputs/eval_full_ocl/drone_to_satellite \
  --device cuda:0 --batch-size 8
```

指标保存在 `OCL_plus/outputs/eval_full_ocl/drone_to_satellite/metrics.json`。

### 多天气测试

University-1652 上评测 9 种天气：`fog`、`rain`、`snow`、`dark`、`light`、`fog_rain`、`fog_snow`、`rain_snow`、`wind`。仅增强无人机查询图像，卫星图库保持原样；增强在线执行，不保存天气图像。

```bash
python OCL_plus/code/eval_ocl_weather.py \
  --checkpoint OCL_plus/outputs/train_full_ocl/checkpoints/best_image.pt \
  --expected-variant mean_teacher \
  --dataset-root ../Datasets/U1652 \
  --output-dir OCL_plus/outputs/weather_full_ocl \
  --device cuda:0 --batch-size 8 --workers 8
```

指定天气可添加 `--weather fog rain snow`；跳过已完成的天气可添加 `--resume`。
汇总结果保存在 `OCL_plus/outputs/weather_full_ocl/weather_summary.csv`，各天气指标位于 `drone_to_satellite/<天气>/metrics.json`。
