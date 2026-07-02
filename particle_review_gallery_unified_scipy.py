"""
Tiled Particle Detection Gallery
Detects particles in pre-made tiles, deduplicates, and shows in interactive gallery

Features:
- Load tiles from directory (auto-reads manifest.json or sidecar JSONs)
- Detect particles in all tiles
- Deduplicate overlapping detections
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
from datetime import datetime
from ultralytics import YOLO
from copy import deepcopy
from tile_particle_manager import TileParticleManager

st.set_page_config(page_title="Tiled Particle Detection", layout="wide")
st.title("🧹 Tiled Particle Detection Gallery")

# CONFIG
MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299
BLACK_BG_THRESHOLD = 30

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


def load_tile_metadata(tiles_dir):
    """Load tile metadata from manifest.json"""
    manifest_path = os.path.join(tiles_dir, "manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest.get("tiles", [])
    else:
        return None


def detect_particles_in_tiles(tiles_dir, tile_metadata, model):
    """Detect particles in all tiles"""
    all_tile_particles = []
    progress_bar = st.progress(0)
    status = st.empty()

    for idx, tile_meta in enumerate(tile_metadata):
        tile_filename = tile_meta["filename"]
        tile_id = idx

        status.text(f"Processing {idx + 1}/{len(tile_metadata)}: {tile_filename}")

        tile_path = os.path.join(tiles_dir, tile_filename)

        if not os.path.exists(tile_path):
            st.warning(f"Tile not found: {tile_path}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        # Load tile
        try:
            img_pil = Image.open(tile_path)
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            tile_img = np.array(img_pil)
            tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.warning(f"Failed to load {tile_filename}: {e}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        # Detect
        try:
            results = model(tile_img, iou=0.45, conf=0.02, verbose=False)
        except Exception as e:
            st.warning(f"Detection failed on {tile_filename}: {e}")
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

                particle = {
                    "tile_id": tile_id,
                    "tile_filename": tile_filename,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "diameter_um": max(x2 - x1, y2 - y1) * CALIBRATION_UM_PER_PIXEL,
                    "mask": mask
                }

                all_tile_particles.append(particle)

        progress_bar.progress((idx + 1) / len(tile_metadata))

    status.empty()
    return all_tile_particles


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
if "tile_images_cache" not in st.session_state:
    st.session_state.tile_images_cache = {}


def push_undo():
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Tiles Directory")

    tiles_dir = st.text_input("Path to tiles:", "/path/to/tiles")

    if st.button("📋 Load Metadata"):
        if os.path.exists(tiles_dir):
            tile_metadata = load_tile_metadata(tiles_dir)
            if tile_metadata:
                st.session_state.tile_metadata = tile_metadata
                st.success(f"✅ Loaded {len(tile_metadata)} tiles")
            else:
                st.error("No manifest.json found")
        else:
            st.error(f"Directory not found: {tiles_dir}")

    st.divider()

    if st.session_state.tile_metadata:
        st.header("🔍 Detection")

        if st.button("Run Inference"):
            model = load_model()
            if model is None:
                st.error("Model not found")
            else:
                # Detect in all tiles
                tile_particles = detect_particles_in_tiles(
                    tiles_dir, st.session_state.tile_metadata, model
                )
                st.success(f"Found {len(tile_particles)} raw detections")

                # Deduplicate
                st.write("Deduplicating...")

                # Save metadata temporarily for TileParticleManager
                import tempfile

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    # Convert to TileParticleManager format
                    mgr_metadata = []
                    for i, tm in enumerate(st.session_state.tile_metadata):
                        mgr_metadata.append({
                            "id": i,
                            "filename": tm["filename"],
                            "x_start": tm["x"],
                            "y_start": tm["y"],
                            "x_end": tm["x"] + tm["width"],
                            "y_end": tm["y"] + tm["height"],
                            "neighbors": tm.get("neighbors", {})
                        })
                    json.dump(mgr_metadata, f)
                    metadata_file = f.name

                try:
                    manager = TileParticleManager(metadata_file, iou_threshold=0.3, seam_margin=30)
                    deduplicated, stats = manager.process_tile_particles(tile_particles)

                    # Store results
                    st.session_state.results = deduplicated
                    st.session_state.undo_stack = []
                    st.session_state.selected_particles = set()

                    st.success(f"✅ {stats['after_dedup']} particles after deduplication")
                    st.info(f"Removed: {stats['duplicates_removed']} | At seams: {stats['at_seams']}")
                except Exception as e:
                    st.error(f"Deduplication failed: {e}")

    st.divider()

    if st.session_state.undo_stack:
        if st.button("↶ Undo"):
            st.session_state.results = st.session_state.undo_stack.pop()
            st.session_state.selected_particles = set()
            st.rerun()

    if st.session_state.results:
        total = len([p for p in st.session_state.results if not p.get("deleted", False)])
        st.success(f"✅ {total} particles")
        st.write(f"**Selected:** {len(st.session_state.selected_particles)}")

    st.divider()

    if st.button("📥 Export CSV"):
        if st.session_state.results:
            rows = []
            for p in st.session_state.results:
                if not p.get("deleted", False):
                    rows.append({
                        "tile": p["tile_filename"],
                        "class": p["class"],
                        "diameter_um": p["diameter_um"],
                        "confidence": round(p["confidence"], 3),
                        "at_seam": p.get("at_seam", False),
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
    st.info("👈 Load metadata and run inference")
else:
    # Filters
    st.subheader("🖼️ Particle Gallery")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_class = st.multiselect(
            "Class:",
            ["Fiber", "Glass", "Metallic", "Other"],
            default=["Fiber", "Glass", "Metallic", "Other"],
            key="fc"
        )
    with col2:
        show_black_only = st.checkbox("Black bg only")
    with col3:
        items_per_page = st.selectbox("Per page:", [12, 18, 24, 36], index=0)

    # Collect particles
    all_particles = []
    for idx, p in enumerate(st.session_state.results):
        if not p.get("deleted", False):
            if p["class"] in filter_class:
                all_particles.append((idx, p))

    if all_particles:
        st.success(f"{len(all_particles)} particles")

        # Pagination
        total_pages = max(1, (len(all_particles) + items_per_page - 1) // items_per_page)
        if total_pages > 1:
            page = st.slider("Page:", 1, total_pages, 1) - 1
        else:
            page = 0

        start_idx = page * items_per_page
        end_idx = start_idx + items_per_page
        page_particles = all_particles[start_idx:end_idx]

        # Gallery
        cols = st.columns(6)
        for i, (pidx, p) in enumerate(page_particles):
            with cols[i % 6]:
                # Load tile image
                tile_filename = p["tile_filename"]
                if tile_filename not in st.session_state.tile_images_cache:
                    tile_path = os.path.join(tiles_dir, tile_filename)
                    tile_img = Image.open(tile_path).convert('RGB')
                    st.session_state.tile_images_cache[tile_filename] = np.array(tile_img)

                tile_np = st.session_state.tile_images_cache[tile_filename]

                # Crop
                x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                margin = 15
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(tile_np.shape[1], x + w + margin)
                y2 = min(tile_np.shape[0], y + h + margin)

                crop = tile_np[y1:y2, x1:x2].copy()

                # Draw box
                crop_pil = Image.fromarray(crop.astype(np.uint8)).convert('RGB')
                draw = ImageDraw.Draw(crop_pil)

                bx1 = x - x1
                by1 = y - y1
                bx2 = x + w - x1
                by2 = y + h - y1

                draw.rectangle([(bx1, by1), (bx2, by2)], outline=(0, 255, 0), width=2)
                crop = np.array(crop_pil)

                # Display
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
            action = st.radio(
                "Action:",
                ["Delete", "Change Class"],
                horizontal=True
            )
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
            st.success(f"✅ Applied to {len(st.session_state.selected_particles)} particles")
            st.rerun()