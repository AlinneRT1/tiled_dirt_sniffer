"""
Tiled Particle Detection Gallery - FULL FEATURED
All features: summary table, gallery, full image zoom, individual edits,
mass edit, undo, sizing method display, bounding boxes

Features:
- Summary table (class × size bin)
- 6-column gallery with pagination
- Green bounding boxes on previews
- Sizing method display (edge_detect, mask_bounds, bbox)
- Full image zoom/pan with Plotly
- Individual class editing
- Delete individual particles
- Select + mass edit
- Undo stack
- CSV export
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json
import os
import tempfile
from datetime import datetime
from ultralytics import YOLO
from copy import deepcopy
import plotly.graph_objects as go
from scipy import ndimage

st.set_page_config(page_title="tiled dirt sniffer", page_icon="icon.ico", layout="wide")
st.title("🐕 tiled_dirt_sniffer: Review Dashboard")

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

def detect_particles_in_tiles(tile_files, tile_metadata, model):
    """Detect in all tiles (loads from uploaded files)"""
    all_particles = []
    progress_bar = st.progress(0)
    status = st.empty()

    for idx, tile_meta in enumerate(tile_metadata):
        filename = tile_meta['filename']
        status.text(f"Detecting {idx + 1}/{len(tile_metadata)}: {filename}")

        # Load from uploaded file
        if filename not in tile_files:
            st.warning(f"Missing: {filename}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        try:
            file_obj = tile_files[filename]
            img_pil = Image.open(file_obj)
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            tile_img = np.array(img_pil)
        except Exception as e:
            st.warning(f"Failed to load {filename}: {e}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        # Convert RGB to BGR for YOLO
        tile_img_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)

        # Detect
        try:
            results = model(tile_img_bgr, iou=0.45, conf=0.02, verbose=False)
        except Exception as e:
            st.warning(f"Detection failed on {filename}: {e}")
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
                    "tile_filename": filename,
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
if "tile_files" not in st.session_state:
    st.session_state.tile_files = {}

def push_undo():
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📤 Upload Tiles")

    st.write("**Step 1: Upload manifest.json**")
    manifest_file = st.file_uploader("Manifest:", type=["json"], key="manifest")

    st.write("**Step 2: Upload tile images**")
    tile_files = st.file_uploader(
        "Tile images:",
        type=["jpg", "jpeg", "png", "tif"],
        accept_multiple_files=True,
        key="tiles"
    )

    if manifest_file and tile_files and st.button("📋 Load"):
        try:
            manifest = json.load(manifest_file)
            tile_metadata = manifest.get("tiles", [])

            file_map = {f.name: f for f in tile_files}

            st.session_state.tile_metadata = tile_metadata
            st.session_state.tile_files = file_map

            st.success(f"✅ Ready to detect!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    if st.session_state.tile_metadata:
        if st.button("🔍 Run Inference"):
            model = load_model()
            if model is None:
                st.error("Model not found")
            else:
                # Step 1: Detect in all tiles
                raw_particles = detect_particles_in_tiles(
                    st.session_state.tile_files,
                    st.session_state.tile_metadata,
                    model
                )
                st.write(f"Raw detections: {len(raw_particles)}")

                # Step 2: Deduplicate using TileParticleManager
                try:
                    from tile_particle_manager import TileParticleManager
                    import tempfile

                    st.write("Deduplicating...")

                    # Convert metadata to TileParticleManager format
                    mgr_metadata = []
                    for i, tm in enumerate(st.session_state.tile_metadata):
                        mgr_metadata.append({
                            "id": i,
                            "filename": tm["filename"],
                            "x_start": tm.get("x", 0),
                            "y_start": tm.get("y", 0),
                            "x_end": tm.get("x", 0) + tm.get("width", 3000),
                            "y_end": tm.get("y", 0) + tm.get("height", 3000),
                            "neighbors": tm.get("neighbors", {})
                        })

                    # Save temp metadata for manager
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(mgr_metadata, f)
                        metadata_file = f.name

                    # Deduplicate
                    manager = TileParticleManager(metadata_file, iou_threshold=0.3, seam_margin=30)
                    dedup_particles, stats = manager.process_tile_particles(raw_particles)

                    st.write(f"✅ After dedup: {stats['after_dedup']}")
                    st.write(f"   Removed: {stats['duplicates_removed']}")
                    st.write(f"   At seams: {stats['at_seams']}")

                    st.session_state.results = dedup_particles

                except ImportError:
                    st.warning("TileParticleManager not found, using raw detections")
                    st.session_state.results = raw_particles
                except Exception as e:
                    st.error(f"Deduplication error: {e}")
                    st.write("Using raw detections")
                    st.session_state.results = raw_particles

                st.session_state.undo_stack = []
                st.session_state.selected_particles = set()
                st.success(f"Done!")

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
    st.info("👈 Upload tiles and run inference")
else:
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("📊 Summary Table")

    data = {}
    for cls in ["Fiber", "Glass", "Metallic", "Other"]:
        data[cls] = {}
        for b, _, _ in SIZE_BINS:
            count = len([p for p in st.session_state.results
                        if p["class"] == cls and p["size_bin"] == b and not p.get("deleted")])
            data[cls][b] = count

    rows = []
    for cls in ["Fiber", "Glass", "Metallic", "Other"]:
        row = {"Material": cls}
        total = 0
        for b, _, _ in SIZE_BINS:
            c = data[cls][b]
            row[b] = c
            total += c
        row["Total"] = total
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=150)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # GALLERY
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("🖼️ Particle Gallery")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_class = st.multiselect(
            "Class:",
            ["Fiber", "Glass", "Metallic", "Other"],
            default=["Fiber", "Glass", "Metallic", "Other"],
            key="fc"
        )
    with col2:
        filter_bins = st.multiselect(
            "Size Bin:",
            [b[0] for b in SIZE_BINS],
            default=[b[0] for b in SIZE_BINS],
            key="fb"
        )
    with col3:
        show_seams_only = st.checkbox("Seams only")
    with col4:
        items_per_page = st.selectbox("Per page:", [12, 18, 24, 36], index=0)

    # Filter particles
    all_particles = []
    for idx, p in enumerate(st.session_state.results):
        if not p.get("deleted") and p["class"] in filter_class and p["size_bin"] in filter_bins:
            if show_seams_only and not p.get("at_seam"):
                continue
            all_particles.append((idx, p))

    if all_particles:
        st.success(f"{len(all_particles)} particles")

        # Pagination
        total_pages = max(1, (len(all_particles) + items_per_page - 1) // items_per_page)
        if total_pages > 1:
            page = st.slider("Page:", 1, total_pages, 1) - 1
        else:
            page = 0

        start = page * items_per_page
        end = start + items_per_page
        page_particles = all_particles[start:end]

        # Gallery
        cols = st.columns(6)
        for i, (pidx, p) in enumerate(page_particles):
            with cols[i % 6]:
                # Load tile
                filename = p["tile_filename"]
                if filename not in st.session_state.tile_files:
                    st.warning("Tile missing")
                    continue

                try:
                    file_obj = st.session_state.tile_files[filename]
                    tile_img = Image.open(file_obj).convert('RGB')
                    tile_img = np.array(tile_img)
                except Exception as e:
                    st.warning(f"Load error")
                    continue

                # Crop
                x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                margin = 15
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(tile_img.shape[1], x + w + margin)
                y2 = min(tile_img.shape[0], y + h + margin)

                crop = tile_img[y1:y2, x1:x2].copy()

                # Draw bright blue box
                crop_pil = Image.fromarray(crop).convert('RGB')
                draw = ImageDraw.Draw(crop_pil)
                draw.rectangle([(x-x1, y-y1), (x+w-x1, y+h-y1)], outline=(0, 100, 255), width=2)
                crop = np.array(crop_pil)

                # Display
                st.image(crop, use_column_width=True)

                # Caption with size bin and sizing method
                method = p["size_method"]
                method_icon = {"edge_detect": "✨", "mask_bounds": "📊", "bbox": "📦"}

                caption = f"{p['class']} | {p['size_bin']}\n{p['diameter_um']:.1f}µm\n({method})"

                # Add seam warning if applicable
                if p.get("at_seam"):
                    caption += f"\n⚠️ At seams: {p.get('seams', [])}"

                st.caption(caption)

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

                # View full
                if st.button("🔍 View Full", key=f"view_{pidx}"):
                    st.session_state[f"show_full_{pidx}"] = True

        # Full image viewer
        for pidx, p in [(idx, p) for idx, p in page_particles]:
            if st.session_state.get(f"show_full_{pidx}", False):
                filename = p["tile_filename"]

                with st.expander(f"Full Image: {filename}", expanded=True):
                    if filename not in st.session_state.tile_files:
                        st.warning("Tile missing")
                        continue

                    try:
                        file_obj = st.session_state.tile_files[filename]
                        tile_img = Image.open(file_obj).convert('RGB')
                        tile_img = np.array(tile_img)
                    except:
                        st.warning("Load error")
                        continue

                    # Create Plotly figure
                    fig = go.Figure()
                    fig.add_trace(go.Image(z=tile_img, name="Image"))

                    # Highlight particle with bright blue box
                    x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                    fig.add_shape(
                        type="rect",
                        x0=x, y0=y, x1=x + w, y1=y + h,
                        line=dict(color="rgb(0, 100, 255)", width=3)
                    )

                    fig.update_layout(
                        title=f"{filename} | {p['class']} ({p['size_bin']}) {p['diameter_um']}µm [{p['size_method']}]",
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=40),
                        height=600,
                    )
                    fig.update_xaxes(scaleanchor="y", scaleratio=1)
                    fig.update_yaxes(scaleanchor="x", scaleratio=1)

                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Class:** {p['class']}")
                    with col2:
                        st.write(f"**Size:** {p['diameter_um']}µm ({p['size_bin']})")
                    with col3:
                        st.write(f"**Method:** {p['size_method']}")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # MASS EDIT
    # ─────────────────────────────────────────────────────────────────────────

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