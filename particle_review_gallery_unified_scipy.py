"""
Tiled Particle Detection Gallery - LAZY LOADING
Loads tiles on-demand instead of caching all at once
Based on particle_review_gallery_unified.py pattern

Features:
- Upload ZIP with tiles + manifest
- Detect particles (loads tiles one at a time)
- Gallery with 6-column layout
- Mass edit + individual edits
- Expanded zoom/pan view
- CSV export
- Undo
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json
import os
import zipfile
import tempfile
from datetime import datetime
from ultralytics import YOLO
from copy import deepcopy
import plotly.graph_objects as go
from scipy import ndimage

st.set_page_config(page_title="Tiled Particle Detection", layout="wide")
st.title("🧹 Tiled Particle Detection Gallery")

# CONFIG
MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299

SIZE_BINS = [
    ("B: 5-15μm", 5, 15),
    ("C: 15-25μm", 15, 25),
    ("D: 25-50μm", 25, 50),
    ("E: 50-100μm", 50, 100),
    ("F: 100-250μm", 100, 250),
    ("G: 250-500μm", 250, 500),
    ("H: 500-750μm", 500, 750),
    ("I: 750-1000μm", 750, 1000),
    ("J: 1000μm+", 1000, float("inf")),
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)

def get_size_bin(diameter_um):
    for label, lo, hi in SIZE_BINS:
        if lo <= diameter_um < hi:
            return label
    return "K"

def calculate_particle_size_accurate(mask_array, calibration):
    """Edge detection sizing"""
    try:
        if mask_array is None or np.sum(mask_array) == 0:
            raise ValueError("Empty mask")
        edges = ndimage.sobel(mask_array.astype(float))
        edge_pixels = np.where(edges > 0.1)
        if len(edge_pixels[0]) > 0:
            y_min, y_max = edge_pixels[0].min(), edge_pixels[0].max()
            x_min, x_max = edge_pixels[1].min(), edge_pixels[1].max()
            diameter_pixels = max(x_max - x_min + 1, y_max - y_min + 1)
            return round(diameter_pixels * calibration, 1), "edge_detect"
    except:
        pass

    try:
        mask_pixels = np.where(mask_array > 0.5)
        if len(mask_pixels[0]) > 0:
            y_min, y_max = mask_pixels[0].min(), mask_pixels[0].max()
            x_min, x_max = mask_pixels[1].min(), mask_pixels[1].max()
            diameter_pixels = max(x_max - x_min + 1, y_max - y_min + 1)
            return round(diameter_pixels * calibration, 1), "mask_bounds"
    except:
        pass

    return None, "failed"

def load_tile_image(tile_dir, tile_filename):
    """Load tile on-demand (lazy loading)"""
    tile_path = os.path.join(tile_dir, tile_filename)
    if os.path.exists(tile_path):
        img_pil = Image.open(tile_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        return np.array(img_pil)
    return None

def detect_particles_in_tiles(tile_dir, tile_metadata, model):
    """Detect in all tiles (loads one at a time)"""
    all_particles = []
    progress_bar = st.progress(0)
    status = st.empty()

    for idx, tile_meta in enumerate(tile_metadata):
        status.text(f"Detecting {idx + 1}/{len(tile_metadata)}: {tile_meta['filename']}")

        tile_img = load_tile_image(tile_dir, tile_meta['filename'])
        if tile_img is None:
            st.warning(f"Failed to load {tile_meta['filename']}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        # Convert RGB to BGR for YOLO
        tile_img_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)

        # Detect
        try:
            results = model(tile_img_bgr, iou=0.45, conf=0.02, verbose=False)
        except Exception as e:
            st.warning(f"Detection failed on {tile_meta['filename']}: {e}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        # Extract particles
        for r in results:
            if r.boxes is None:
                continue

            for i, (box, cls, conf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]

                # Get mask
                try:
                    mask = r.masks.data[i].cpu().numpy() if hasattr(r.masks.data[i], 'cpu') else r.masks.data[i]
                except:
                    mask = None

                diameter_um, method = calculate_particle_size_accurate(mask, CALIBRATION_UM_PER_PIXEL)
                if diameter_um is None:
                    diameter_um = max(x2 - x1, y2 - y1) * CALIBRATION_UM_PER_PIXEL
                    method = "bbox"

                all_particles.append({
                    "tile_id": idx,
                    "tile_filename": tile_meta['filename'],
                    "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "diameter_um": diameter_um,
                    "size_bin": get_size_bin(diameter_um),
                    "size_method": method,
                    "deleted": False
                })

        progress_bar.progress((idx + 1) / len(tile_metadata))

    status.empty()
    return all_particles

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "selected_particles" not in st.session_state:
    st.session_state.selected_particles = set()
if "tile_metadata" not in st.session_state:
    st.session_state.tile_metadata = None
if "tile_dir_path" not in st.session_state:
    st.session_state.tile_dir_path = None

def push_undo():
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📤 Upload Tiles")

    zip_file = st.file_uploader("Upload tiles.zip", type=["zip"])

    if zip_file and st.button("📋 Extract & Load"):
        try:
            # Extract to persistent temp dir
            tmpdir = tempfile.mkdtemp()
            st.session_state.tmpdir = tmpdir

            with zipfile.ZipFile(zip_file) as z:
                z.extractall(tmpdir)

            st.write("✅ Extracted")

            # Find manifest
            manifest_path = None
            for root, dirs, files in os.walk(tmpdir):
                if "manifest.json" in files:
                    manifest_path = os.path.join(root, "manifest.json")
                    break

            if manifest_path:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                tile_metadata = manifest.get("tiles", [])
                st.session_state.tile_metadata = tile_metadata
                st.session_state.tile_dir_path = os.path.dirname(manifest_path)

                st.success(f"✅ Loaded {len(tile_metadata)} tiles")
            else:
                st.error("manifest.json not found")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    if st.session_state.tile_metadata:
        if st.button("🔍 Run Inference"):
            model = load_model()
            if model is None:
                st.error("Model not found")
            else:
                particles = detect_particles_in_tiles(
                    st.session_state.tile_dir_path,
                    st.session_state.tile_metadata,
                    model
                )
                st.session_state.results = particles
                st.session_state.undo_stack = []
                st.session_state.selected_particles = set()
                st.success(f"Found {len(particles)} particles")

    st.divider()

    if st.session_state.undo_stack:
        if st.button("↶ Undo"):
            st.session_state.results = st.session_state.undo_stack.pop()
            st.session_state.selected_particles = set()
            st.rerun()

    if st.session_state.results:
        total = len([p for p in st.session_state.results if not p.get("deleted")])
        st.success(f"✅ {total} particles")
        st.write(f"**Selected:** {len(st.session_state.selected_particles)}")

    st.divider()

    if st.button("📥 Export CSV"):
        if st.session_state.results:
            rows = []
            for p in st.session_state.results:
                if not p.get("deleted"):
                    rows.append({
                        "tile": p["tile_filename"],
                        "class": p["class"],
                        "diameter_um": p["diameter_um"],
                        "size_method": p["size_method"],
                        "confidence": round(p["confidence"], 3),
                    })

            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download",
                csv,
                f"particles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.results is None:
    st.info("👈 Upload ZIP and run inference")
else:
    st.subheader("🖼️ Particle Gallery")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_class = st.multiselect(
            "Class:",
            ["Fiber", "Glass", "Metallic", "Other"],
            default=["Fiber", "Glass", "Metallic", "Other"],
            key="fc"
        )
    with col2:
        items_per_page = st.selectbox("Per page:", [12, 18, 24, 36], index=0)

    # Filter particles
    all_particles = []
    for idx, p in enumerate(st.session_state.results):
        if not p.get("deleted") and p["class"] in filter_class:
            all_particles.append((idx, p))

    if all_particles:
        st.success(f"{len(all_particles)} particles")

        # Pagination
        total_pages = max(1, (len(all_particles) + items_per_page - 1) // items_per_page)
        page = st.slider("Page:", 1, total_pages, 1) - 1

        start = page * items_per_page
        end = start + items_per_page
        page_particles = all_particles[start:end]

        # Gallery
        cols = st.columns(6)
        for i, (pidx, p) in enumerate(page_particles):
            with cols[i % 6]:
                # Load tile on-demand
                tile_img = load_tile_image(st.session_state.tile_dir_path, p["tile_filename"])
                if tile_img is None:
                    st.warning("Tile load failed")
                    continue

                # Crop
                x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                margin = 15
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(tile_img.shape[1], x + w + margin)
                y2 = min(tile_img.shape[0], y + h + margin)

                crop = tile_img[y1:y2, x1:x2].copy()

                # Draw box
                crop_pil = Image.fromarray(crop).convert('RGB')
                draw = ImageDraw.Draw(crop_pil)
                draw.rectangle([(x-x1, y-y1), (x+w-x1, y+h-y1)], outline=(0, 255, 0), width=2)
                crop = np.array(crop_pil)

                st.image(crop, use_column_width=True)
                st.caption(f"{p['class']}\n{p['diameter_um']:.1f}µm")

                # Checkbox
                key = f"sel_{pidx}"
                is_selected = key in st.session_state.selected_particles
                if st.checkbox("Select", value=is_selected, key=key):
                    st.session_state.selected_particles.add(key)
                else:
                    st.session_state.selected_particles.discard(key)

                # Edit class
                new_cls = st.selectbox(
                    "Class:",
                    ["Fiber", "Glass", "Metallic", "Other"],
                    index=["Fiber", "Glass", "Metallic", "Other"].index(p["class"]),
                    key=f"cls_{pidx}"
                )
                if new_cls != p["class"] and st.button("✓", key=f"save_{pidx}"):
                    push_undo()
                    st.session_state.results[pidx]["class"] = new_cls
                    st.rerun()

                # Delete
                if st.button("🗑️", key=f"del_{pidx}"):
                    push_undo()
                    st.session_state.results[pidx]["deleted"] = True
                    st.rerun()

    st.divider()

    # Mass edit
    if st.session_state.selected_particles:
        st.subheader("⚙️ Bulk Edit")

        col1, col2 = st.columns(2)
        with col1:
            action = st.radio("Action:", ["Delete", "Change Class"], horizontal=True)
        with col2:
            if action == "Change Class":
                new_cls = st.selectbox("To:", ["Fiber", "Glass", "Metallic", "Other"])

        if st.button("Execute"):
            push_undo()
            for key in st.session_state.selected_particles:
                pidx = int(key.split("_")[1])
                if action == "Delete":
                    st.session_state.results[pidx]["deleted"] = True
                else:
                    st.session_state.results[pidx]["class"] = new_cls

            st.session_state.selected_particles = set()
            st.success(f"✅ Done")
            st.rerun()