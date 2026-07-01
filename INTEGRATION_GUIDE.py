"""
How to integrate TileParticleManager into your existing particle detection app

ADD THIS to your existing detection code:
"""

from tile_particle_manager import TileParticleManager
import json

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def process_tiles_with_deduplication(tiles_directory, metadata_file, model):
    """
    Load tiles from disk and detect particles with deduplication
    
    Args:
        tiles_directory: folder with tile_000.png, tile_001.png, etc.
        metadata_file: tile_metadata.json with positions and neighbors
        model: YOLO model
    
    Returns:
        deduplicated_particles, stats
    """
    
    # Step 1: Load metadata
    with open(metadata_file, 'r') as f:
        tile_metadata = json.load(f)
    
    st.write(f"Loaded metadata for {len(tile_metadata)} tiles")
    
    # Step 2: Detect in each tile
    tile_particles = []
    progress_bar = st.progress(0)
    
    for idx, tile_meta in enumerate(tile_metadata):
        # Load tile image
        tile_path = f"{tiles_directory}/{tile_meta['filename']}"
        tile_img = cv2.imread(tile_path)
        
        # Detect
        results = model(tile_img, iou=0.45, conf=0.02, verbose=False)
        
        # Collect particles with tile_id
        for r in results:
            if r.boxes is None:
                continue
            
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                
                particle = {
                    "tile_id": tile_meta["id"],
                    "x": x1,  # tile-local coords
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "diameter_um": max(x2 - x1, y2 - y1) * CALIBRATION_UM_PER_PIXEL,
                }
                
                tile_particles.append(particle)
        
        progress_bar.progress((idx + 1) / len(tile_metadata))
    
    st.success(f"Detected {len(tile_particles)} particles in tiles")
    
    # Step 3: Deduplicate using TileParticleManager
    st.write("⚙️ Deduplicating and marking cut particles...")
    
    manager = TileParticleManager(
        metadata_file=metadata_file,
        iou_threshold=0.3,
        seam_margin=30
    )
    
    deduplicated, stats = manager.process_tile_particles(tile_particles)
    
    return deduplicated, stats


# ─────────────────────────────────────────────────────────────────────────────
# ADD THIS TO YOUR STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

# In your sidebar or main section:

st.header("🧹 Tiled Particle Detection")

col1, col2 = st.columns(2)

with col1:
    tiles_dir = st.text_input("Tiles directory path", "/path/to/tiles")

with col2:
    metadata_file = st.text_input("Metadata JSON file", "/path/to/tile_metadata.json")

if st.button("🔍 Detect & Deduplicate"):
    model = load_model()
    
    try:
        # Run detection with deduplication
        deduplicated, stats = process_tiles_with_deduplication(
            tiles_dir, metadata_file, model
        )
        
        # Display stats
        st.markdown("### 📊 Deduplication Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Raw Detections", stats["total_input"])
        with col2:
            st.metric("After Dedup", stats["after_dedup"])
        with col3:
            st.metric("Duplicates Removed", stats["duplicates_removed"])
        with col4:
            st.metric("At Seams", stats["at_seams"])
        
        # Display results
        st.markdown("### 🎯 Particles")
        
        results_data = []
        for p in deduplicated:
            results_data.append({
                "Tile": p["tile_id"],
                "Class": p["class"],
                "Size (µm)": p["diameter_um"],
                "Confidence": round(p["confidence"], 3),
                "At Seam": "⚠️" if p["at_seam"] else "",
                "Seams": ", ".join(p["seams"]) if p["seams"] else "-",
                "Mosaic X": p["mosaic_x"],
                "Mosaic Y": p["mosaic_y"],
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
        
        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "particles_deduplicated.csv",
            "text/csv"
        )
        
        # Highlight cut particles
        cut_particles = [p for p in deduplicated if p["at_seam"]]
        if cut_particles:
            st.warning(f"⚠️ {len(cut_particles)} particles at tile seams (may be cut)")
    
    except Exception as e:
        st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# THAT'S IT!
# ─────────────────────────────────────────────────────────────────────────────
# 
# Just add those 3 sections to your existing app:
# 1. process_tiles_with_deduplication() function
# 2. Streamlit UI section (sidebar or main)
# 3. Done!
#
# The TileParticleManager handles all the dedup + cut detection
