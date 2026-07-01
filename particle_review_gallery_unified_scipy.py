"""
Tiled Particle Detection - DEBUG VERSION
Better error messages to find the problem
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import traceback
from tile_particle_manager import TileParticleManager

st.set_page_config(page_title="Tiled Detection - Debug", layout="wide")
st.title("🧹 Tiled Particle Detection (Debug)")

MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299

@st.cache_resource
def load_model():
    st.write("Loading model...")
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        st.success(f"✅ Model loaded: {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        st.write(traceback.format_exc())
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

st.header("📂 Input Paths")

col1, col2 = st.columns(2)

with col1:
    tiles_dir = st.text_input("Tiles directory:", "/home/user/tiles")

with col2:
    metadata_file = st.text_input("Metadata JSON:", "/home/user/tile_metadata.json")

# Validate paths
st.header("✅ Validation")

path_ok = True

if not os.path.exists(tiles_dir):
    st.error(f"❌ Tiles dir not found: {tiles_dir}")
    path_ok = False
else:
    tile_files = [f for f in os.listdir(tiles_dir) if f.endswith(('.png', '.jpg', '.tif'))]
    st.success(f"✅ Found {len(tile_files)} tile files")
    if tile_files:
        st.write(f"  Example: {tile_files[0]}")

if not os.path.exists(metadata_file):
    st.error(f"❌ Metadata file not found: {metadata_file}")
    path_ok = False
else:
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        st.success(f"✅ Loaded metadata for {len(metadata)} tiles")
    except Exception as e:
        st.error(f"❌ Metadata parse error: {e}")
        path_ok = False

if not path_ok:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RUN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

if st.button("🔍 Run Inference (Debug)"):
    st.header("Running Detection...")

    try:
        # Step 1: Load model
        st.subheader("Step 1️⃣: Loading Model")
        model = load_model()
        if model is None:
            st.stop()

        # Step 2: Load metadata
        st.subheader("Step 2️⃣: Loading Metadata")
        with open(metadata_file) as f:
            tile_metadata = json.load(f)
        st.success(f"✅ Loaded {len(tile_metadata)} tiles")

        # Step 3: Detect in each tile
        st.subheader("Step 3️⃣: Running Detection on Tiles")

        tile_particles = []
        progress_bar = st.progress(0)
        status = st.empty()

        for idx, tile_meta in enumerate(tile_metadata):
            tile_id = tile_meta["id"]
            tile_filename = tile_meta["filename"]

            status.text(f"Processing tile {idx + 1}/{len(tile_metadata)}: {tile_filename}")

            try:
                # Load tile
                tile_path = os.path.join(tiles_dir, tile_filename)

                if not os.path.exists(tile_path):
                    st.warning(f"⚠️ Tile not found: {tile_path}")
                    progress_bar.progress((idx + 1) / len(tile_metadata))
                    continue

                # Try PIL first (better for PNG)
                try:
                    img_pil = Image.open(tile_path)
                    if img_pil.mode != 'RGB':
                        img_pil = img_pil.convert('RGB')
                    tile_img = np.array(img_pil)
                    # Convert RGB → BGR for YOLO
                    tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
                except:
                    # Fallback to cv2
                    tile_img = cv2.imread(tile_path)

                if tile_img is None:
                    st.warning(f"⚠️ Failed to load: {tile_filename}")
                    progress_bar.progress((idx + 1) / len(tile_metadata))
                    continue

                # Run YOLO detection
                try:
                    results = model(tile_img, iou=0.45, conf=0.02, verbose=False)
                except Exception as e:
                    st.warning(f"⚠️ Detection failed on {tile_filename}: {e}")
                    progress_bar.progress((idx + 1) / len(tile_metadata))
                    continue

                # Extract particles
                for r in results:
                    if r.boxes is None:
                        continue

                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]

                        particle = {
                            "tile_id": tile_id,
                            "x": x1,
                            "y": y1,
                            "w": x2 - x1,
                            "h": y2 - y1,
                            "class": model.names[int(cls)],
                            "confidence": float(conf),
                            "diameter_um": max(x2 - x1, y2 - y1) * CALIBRATION_UM_PER_PIXEL,
                        }

                        tile_particles.append(particle)

            except Exception as e:
                st.error(f"❌ Error on tile {idx}: {e}")
                st.write(traceback.format_exc())

            progress_bar.progress((idx + 1) / len(tile_metadata))

        status.empty()
        st.success(f"✅ Detected {len(tile_particles)} particles")

        # Step 4: Deduplicate
        st.subheader("Step 4️⃣: Deduplicating")

        try:
            manager = TileParticleManager(metadata_file, iou_threshold=0.3, seam_margin=30)
            deduplicated, stats = manager.process_tile_particles(tile_particles)

            st.success(f"✅ Deduplication complete")

            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Raw Detections", stats["total_input"])
            with col2:
                st.metric("After Dedup", stats["after_dedup"])
            with col3:
                st.metric("Removed", stats["duplicates_removed"])
            with col4:
                st.metric("At Seams", stats["at_seams"])

            # Display results
            st.subheader("Results Table")

            import pandas as pd

            df_data = []
            for p in deduplicated:
                df_data.append({
                    "Tile": p["tile_id"],
                    "Class": p["class"],
                    "Size (µm)": p["diameter_um"],
                    "Confidence": round(p["confidence"], 3),
                    "At Seam": "⚠️" if p["at_seam"] else "",
                    "Mosaic X": p["mosaic_x"],
                    "Mosaic Y": p["mosaic_y"],
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "particles_deduplicated.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"❌ Deduplication failed: {e}")
            st.write(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Fatal error: {e}")
        st.write(traceback.format_exc())