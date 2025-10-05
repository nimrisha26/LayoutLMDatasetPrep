#!/usr/bin/env python3
"""
highlight_labels.py
Extended version:
    - Headers (B/I-HEADER): Blue
    - Tables (B/I-TABLE): Green
"""

import os
import json
from collections import defaultdict
from PIL import Image, ImageDraw

# Extended label mapping (inverse of synthetic_labeling)
LABEL_MAP_INV = {
    0: "O",
    1: "B-HEADER",
    2: "I-HEADER",
    3: "B-TABLE",
    4: "I-TABLE",
}

# Color map for visualization
COLOR_MAP = {
    "HEADER": {"outline": (0, 0, 255, 200), "fill": (0, 0, 255, 60)},   # Blue
    "TABLE": {"outline": (0, 180, 0, 200), "fill": (0, 255, 0, 60)},    # Green
}

def denormalize_bbox(norm_bbox, image_size):
    """Convert 0-1000 bbox to pixel coordinates."""
    w, h = image_size
    x0 = int((norm_bbox[0] / 1000) * w)
    y0 = int((norm_bbox[1] / 1000) * h)
    x1 = int((norm_bbox[2] / 1000) * w)
    y1 = int((norm_bbox[3] / 1000) * h)
    return (x0, y0, x1, y1)

def highlight_labels(chunks, outdir):
    os.makedirs(outdir, exist_ok=True)
    by_image = defaultdict(list)
    for c in chunks:
        by_image[c["image_path"]].append(c)

    for image_path, chunk_list in by_image.items():
        pil = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for c in chunk_list:
            labels = c["labels"]
            bboxes = c["bboxes"]

            for lab, bbox in zip(labels, bboxes):
                if lab is None or lab == -100 or lab == 0:
                    continue

                label_name = LABEL_MAP_INV.get(lab, "O")

                # Determine category (HEADER / TABLE)
                if "HEADER" in label_name:
                    color = COLOR_MAP["HEADER"]
                elif "TABLE" in label_name:
                    color = COLOR_MAP["TABLE"]
                else:
                    continue

                rect = denormalize_bbox(bbox, pil.size)
                draw.rectangle(rect, outline=color["outline"], width=2)
                draw.rectangle(rect, fill=color["fill"])

        combined = Image.alpha_composite(pil, overlay)
        out_path = os.path.join(outdir, os.path.basename(image_path))
        combined.convert("RGB").save(out_path, "PNG")
        print(f"✅ Saved highlighted image: {out_path}")
