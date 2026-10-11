"""Take the paper off the Gray's Anatomy plates: ink becomes opacity, so they sit on any background.

Writes small masks per plate (opacity = ink): <plate>_ink.webp and, for the red arteries in plate 1100,
<plate>_red.webp. The site colours them with CSS, so one file serves both themes and switches instantly:
  dark theme:  ink #CECDC3, red #D14D41 (Flexoki dark text and red)
  light theme: the same colours with their lightness flipped (hue kept): ink #3C3B31, red #BE3A2E
Run:  python static/art/cutout.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

for plate in ["gray1100_pancreas", "gray1105_islets"]:
    rgb = np.asarray(Image.open(HERE / f"{plate}.png").convert("RGB")).astype(float)
    lum = rgb @ [0.299, 0.587, 0.114]
    alpha = np.clip((250 - lum) / (250 - 40), 0, 1) ** 0.85          # paper (>= 250) -> clear, black ink -> solid
    red = (rgb[..., 0] - rgb[..., 1:].max(-1)) > 50                   # the coloured arteries
    ink = (np.round(np.where(red, 0, alpha) * 31) * 255 / 31).astype(np.uint8)        # 32 shades: invisible, much smaller
    def save(a, path):                                                 # opacity = ink (CSS mask), WebP: half a PNG
        Image.fromarray(np.stack([np.full_like(a, 255)] * 3 + [a], -1), "RGBA").save(path, "WEBP", quality=85, method=6)
    save(ink, HERE / f"{plate}_ink.webp")
    if red.any():
        save((red * 235).astype(np.uint8), HERE / f"{plate}_red.webp")
    print(plate, red.any())
