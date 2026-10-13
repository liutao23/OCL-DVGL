"""Standalone utility for generating weather-augmented test images."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Any, Dict, Iterable, Optional, Tuple
if 'complex' not in np.__dict__:
    np.complex = complex
if 'bool' not in np.__dict__:
    np.bool = np.bool_
try:
    import imgaug.augmenters as iaa
except ModuleNotFoundError:
    iaa = None
WEATHER_NAMES: Tuple[str, ...] = ('clear', 'fog', 'rain', 'snow', 'dark', 'light', 'fog_rain', 'fog_snow', 'rain_snow', 'wind')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def build_weather_augmenters() -> Dict[str, Optional[Any]]:
    """Return the ten weather conditions used by the project."""
    if iaa is None:
        raise ModuleNotFoundError('Weather augmentation requires imgaug; install it with `pip install imgaug`.')

    def fog():
        return iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.5, seed=35)
    return {'clear': None, 'fog': iaa.Sequential([fog()]), 'rain': iaa.Sequential([iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38), iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35), iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73), iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93), iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95)]), 'snow': iaa.Sequential([iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38), iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35), iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74), iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94), iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96)]), 'dark': iaa.Sequential([iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31), iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991)]), 'light': iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)]), 'fog_rain': iaa.Sequential([fog(), iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35), iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)]), 'fog_snow': iaa.Sequential([fog(), iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35), iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)]), 'rain_snow': iaa.Sequential([iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35), iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35), iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92), iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91), iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74)]), 'wind': iaa.Sequential([iaa.MotionBlur(15, seed=17)])}

def apply_weather(image, weather: str, augmenters=None) -> np.ndarray:
    """Apply one preset to a PIL image or numpy array; return an RGB array."""
    presets = augmenters if augmenters is not None else build_weather_augmenters()
    if weather not in presets:
        raise ValueError(f'Unknown weather {weather!r}; choose from {tuple(presets)}')
    array = np.asarray(image.convert('RGB') if isinstance(image, Image.Image) else image)
    augmenter = presets[weather]
    return array.copy() if augmenter is None else augmenter(image=array)
