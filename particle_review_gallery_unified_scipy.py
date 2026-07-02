"""
Bare-Bones Particle Gallery
Just shows detected particles with masks and bounding boxes drawn on them
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Particle Gallery", layout="wide")
st.title("🧹 Particle Gallery")

MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# Upload
st.header("📤 Upload Images")
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "tif"])

if uploaded_files:
    if st.button("🔍 Detect Particles"):
        model = load_model()

        # Load image
        img_pil = Image.open(uploaded_files)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        img_np = np.array(img_pil)

        st.write(f"Image: {img_np.shape[0]}×{img_np.shape[1]}px")

        # Detect
        results = model(img_np, iou=0.45, conf=0.02, verbose=False)

        particles = []
        for r in results:
            if r.boxes is None or r.masks is None:
                continue

            for i, (box, cls, conf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]

                # Get mask
                try:
                    mask = r.masks.data[i].cpu().numpy() if hasattr(r.masks.data[i], 'cpu') else r.masks.data[i]
                except:
                    mask = None

                particle = {
                    "x": x1, "y": y1, "x2": x2, "y2": y2,
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "diameter_um": max(x2 - x1, y2 - y1) * CALIBRATION_UM_PER_PIXEL,
                    "mask": mask
                }
                particles.append(particle)

        st.success(f"Found {len(particles)} particles")

        # Gallery
        st.header("🖼️ Gallery")

        cols = st.columns(6)
        for i, p in enumerate(particles):
            with cols[i % 6]:
                x1, y1, x2, y2 = p["x"], p["y"], p["x2"], p["y2"]

                # Crop
                margin = 15
                cx1 = max(0, x1 - margin)
                cy1 = max(0, y1 - margin)
                cx2 = min(img_np.shape[1], x2 + margin)
                cy2 = min(img_np.shape[0], y2 + margin)

                crop = img_np[cy1:cy2, cx1:cx2].copy()

                # Draw bounding box
                crop_pil = Image.fromarray(crop.astype(np.uint8)).convert('RGB')
                draw = ImageDraw.Draw(crop_pil)

                # Box coords in crop space
                bx1 = x1 - cx1
                by1 = y1 - cy1
                bx2 = x2 - cx1
                by2 = y2 - cy1

                # Draw green box
                draw.rectangle([(bx1, by1), (bx2, by2)], outline=(0, 255, 0), width=2)

                # Draw mask if available
                if p["mask"] is not None:
                    mask_crop = p["mask"][cy1:cy2, cx1:cx2]

                    # Create cyan overlay for mask
                    overlay = Image.new('RGBA', crop_pil.size, (0, 0, 0, 0))
                    overlay_arr = np.array(overlay)

                    mask_pixels = np.where(mask_crop > 0.5)
                    for my, mx in zip(mask_pixels[0], mask_pixels[1]):
                        overlay_arr[my, mx] = (0, 255, 255, 100)  # Cyan, semi-transparent

                    overlay = Image.fromarray(overlay_arr)
                    crop_pil = Image.alpha_composite(crop_pil.convert('RGBA'), overlay).convert('RGB')

                crop = np.array(crop_pil)

                # Display
                st.image(crop, use_column_width=True)
                st.caption(f"{p['class']}\n{p['diameter_um']:.1f}µm\nConf: {p['confidence']:.2f}")