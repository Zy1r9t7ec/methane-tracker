import requests
import streamlit as st
import sys, json, time, tempfile, subprocess
import numpy as np
import rasterio
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

API_URL = "http://127.0.0.1:5000/predict"

sys.path.insert(0, str(Path(__file__).parent))
from infer import run_inference

# ── Auto-launch Flask endpoint if not already running ──────────
if "_flask_proc" not in st.session_state:
    try:
        requests.get("http://127.0.0.1:5000/", timeout=1)
        st.session_state["_flask_proc"] = None  # already running externally
    except Exception:
        try:
            _endpoint = Path(__file__).parent / "endpoint.py"
            _proc = subprocess.Popen(
                [sys.executable, str(_endpoint)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.5)  # give Flask a moment to bind
            st.session_state["_flask_proc"] = _proc
        except Exception:
            st.session_state["_flask_proc"] = None

st.set_page_config(page_title="MethanePulse", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem; max-width: 1400px; }

  .topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1.2rem; border-bottom: 1px solid #21262d; margin-bottom: 2rem;
  }
  .topnav-brand { font-size: 1.1rem; font-weight: 700; color: #e6edf3; letter-spacing: 0.02em; }
  .topnav-sub   { font-size: 0.78rem; color: #8b949e; margin-left: 0.6rem; }

  .section-label {
    font-size: 0.7rem; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;
  }

  /* Architecture flow cards */
  .arch-wrapper {
    display: flex; flex-direction: column; gap: 0; align-items: stretch;
  }
  .arch-block {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 0.9rem 1.2rem;
  }
  .arch-block-title {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 0.35rem;
  }
  .arch-block-body {
    font-size: 0.8rem; color: #8b949e; line-height: 1.6;
  }
  .arch-arrow {
    text-align: center; color: #30363d; font-size: 1rem;
    margin: 2px 0; line-height: 1;
  }
  .arch-frozen { color: #e03131; }
  .arch-train  { color: #6daa45; }
  .arch-neutral{ color: #4f98a3; }

  /* Config grid */
  .cfg-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.5rem; margin-top: 0.5rem;
  }
  .cfg-item {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 0.6rem 0.9rem;
  }
  .cfg-key   { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; }
  .cfg-val   { font-size: 0.88rem; font-weight: 600; color: #e6edf3; margin-top: 2px; }

  /* Limitation cards */
  .lim-card {
    background: #161b22; border-left: 3px solid #21262d;
    border-radius: 0 6px 6px 0; padding: 0.5rem 0.9rem;
    margin-bottom: 0.4rem; font-size: 0.82rem; color: #8b949e; line-height: 1.5;
  }
  .lim-card b { color: #c9d1d9; }

  /* Status pill */
  .pill-alert {
    background: #1a0000; border: 1px solid #e03131; color: #e03131;
    border-radius: 6px; padding: 0.6rem 1rem; font-weight: 700; font-size: 0.9rem; margin-bottom: 1rem;
  }
  .pill-clear {
    background: #001a0a; border: 1px solid #4f98a3; color: #4f98a3;
    border-radius: 6px; padding: 0.6rem 1rem; font-weight: 700; font-size: 0.9rem; margin-bottom: 1rem;
  }

  .stButton > button {
    background: #01696f !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 600 !important; width: 100% !important;
    padding: 0.55rem 0 !important; font-size: 0.9rem !important;
  }
  .stButton > button:hover { background: #0c4e54 !important; }
  .stButton > button:disabled { background: #21262d !important; color: #4f4f4f !important; }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid #21262d !important; background: transparent !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: #8b949e !important; font-size: 0.85rem !important; padding: 0.5rem 1.2rem !important;
  }
  .stTabs [aria-selected="true"] {
    color: #e6edf3 !important; border-bottom: 2px solid #01696f !important;
  }
  [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; }
  [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; color: #8b949e !important;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
</style>
""", unsafe_allow_html=True)

BASE_DIR   = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "sample_input"
CKPT_PATH  = BASE_DIR / "terramind_methane_best.pt"
RESULTS    = BASE_DIR / "results"
DOCS_DIR   = BASE_DIR / "docs"
SEV_FG     = {"HIGH": "#e03131", "MEDIUM": "#ff8c00", "LOW": "#6daa45"}

def tiff_to_rgb(path):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    eps = 1e-6
    def norm(b):
        lo, hi = np.percentile(b, 2), np.percentile(b, 98)
        return np.clip((b - lo) / (hi - lo + eps), 0, 1)
    r, g = norm(img[11]), norm(img[12])
    return (np.stack([r, g, np.zeros_like(r)], axis=-1) * 255).astype(np.uint8)

def overlay_mask(rgb_np, mask_path, alpha=0.6):
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)
    base    = Image.fromarray(rgb_np).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    heat    = Image.new("RGBA", base.size, (224, 49, 49, int(255 * alpha)))
    overlay.paste(heat, mask=Image.fromarray(mask * 255, mode="L"))
    return Image.alpha_composite(base, overlay).convert("RGB")

def load_history():
    p = DOCS_DIR / "terramind_training_history.json"
    return json.load(open(p)) if p.exists() else []

def load_alerts():
    RESULTS.mkdir(parents=True, exist_ok=True)
    return [json.load(open(f)) for f in sorted(RESULTS.glob("*_alert.json"))]

# ── Top nav
st.markdown("""
<div class="topnav">
  <div>
    <span class="topnav-brand">MethanePulse</span>
    <span class="topnav-sub">Orbital methane detection · TerraMind-1.0-small</span>
  </div>
  <div style="font-size:0.78rem;color:#8b949e">
    Val IoU 0.3257 &nbsp;·&nbsp; 85 MB &nbsp;·&nbsp; 616K params
  </div>
</div>
""", unsafe_allow_html=True)

tab_demo, tab_alerts, tab_model = st.tabs(["Demo", "Alert Log", "Model Card"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — DEMO
# ═══════════════════════════════════════════════════════════════
with tab_demo:
    left, _, right = st.columns([0.9, 0.05, 1.4])

    with left:
        st.markdown('<p class="section-label">Input</p>', unsafe_allow_html=True)
        src_mode = st.radio("Tile source", ["Sample tile", "Upload tile"],
                            horizontal=True, label_visibility="collapsed", key="tile_source_radio")
        tiff_path = None
        if src_mode == "Sample tile":
            samples = list(SAMPLE_DIR.glob("*.tiff"))
            if samples:
                tiff_path = samples[0]
                st.image(tiff_to_rgb(tiff_path), caption="B11 + B12 false colour", width="stretch")
                st.caption(f"`{tiff_path.name}`")
            else:
                st.error("No sample tile found in sample_input/")
        else:
            up = st.file_uploader("Select .tiff", type=["tiff","tif"], label_visibility="collapsed")
            if up:
                tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
                tmp.write(up.read()); tmp.close()
                tiff_path = Path(tmp.name)
                st.image(tiff_to_rgb(tiff_path), caption="B11 + B12 false colour", width="stretch")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        with st.expander("Advanced settings"):
            threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
        if "threshold" not in dir(): threshold = 0.5
        run = st.button("Run Inference", disabled=(tiff_path is None))

    with right:
        st.markdown('<p class="section-label">Results</p>', unsafe_allow_html=True)
        if run and tiff_path:
            progress = st.progress(0, text="Loading model...")

            # Try Flask endpoint first; fall back to direct in-process inference
            packet = None
            try:
                progress.progress(15, text="Trying remote endpoint...")
                with open(str(tiff_path), "rb") as f:
                    response = requests.post(
                        API_URL,
                        files={"file": f},
                        data={"threshold": threshold},
                        timeout=10,
                    )
                if response.ok:
                    packet = response.json()
                    progress.progress(90, text="Received response from endpoint...")
            except Exception:
                packet = None  # Flask not running — fall through to local inference

            if packet is None:
                # Direct in-process inference — no Flask server required
                progress.progress(20, text="Running inference locally...")
                if not CKPT_PATH.exists():
                    progress.empty()
                    st.error(
                        f"Checkpoint not found: `{CKPT_PATH}`. "
                        "Place `terramind_methane_best.pt` next to app.py."
                    )
                    st.stop()
                try:
                    packet = run_inference(
                        tiff_path            = str(tiff_path),
                        checkpoint_path      = str(CKPT_PATH),
                        output_dir           = str(RESULTS),
                        confidence_threshold = threshold,
                    )
                    progress.progress(90, text="Inference complete...")
                except Exception as exc:
                    progress.empty()
                    st.error(f"Inference failed: {exc}")
                    st.stop()

            st.session_state["packet"]    = packet
            st.session_state["tile_path"] = str(tiff_path)
            progress.progress(100, text="Done")
            time.sleep(0.3)
            progress.empty()

        if "packet" in st.session_state:
            pkt = st.session_state["packet"]
            det = pkt["detection"]; sev = pkt["severity"]
            is_alert = pkt["status"] == "ALERT"
            st.markdown(
                f'<div class="{"pill-alert" if is_alert else "pill-clear"}">'
                f'{"METHANE PLUME DETECTED" if is_alert else "TILE CLEAR — NO PLUME"}</div>',
                unsafe_allow_html=True
            )
            img_col, stat_col = st.columns([1.3, 1])
            with img_col:
                stem      = Path(st.session_state["tile_path"]).stem
                mask_path = RESULTS / f"{stem}_mask.tiff"
                rgb_np    = tiff_to_rgb(st.session_state["tile_path"])
                if mask_path.exists() and det["plume_detected"]:
                    st.image(overlay_mask(rgb_np, mask_path), caption="Plume overlay (red = detected)", width="stretch")
                else:
                    st.image(rgb_np, caption="No plume detected", width="stretch")
            with stat_col:
                st.metric("Plume pixels", f"{det['plume_pixel_count']:,}")
                st.metric("Coverage",     f"{det['plume_coverage_pct']}%")
                st.metric("Mean prob",    f"{det['mean_plume_prob']:.3f}")
                sev_color = SEV_FG.get(sev["label"], "#aaa")
                st.markdown(
                    f'<p style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin-bottom:0.2rem">Severity</p>'
                    f'<p style="font-size:1.8rem;font-weight:800;color:{sev_color};margin:0">{sev["label"]}</p>'
                    f'<p style="font-size:0.78rem;color:#8b949e;margin:0">conf {sev["confidence"]:.1%}</p>',
                    unsafe_allow_html=True
                )
                if det.get("centroid_geographic"):
                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                    st.caption(f"Centroid  {det['centroid_geographic']}")
            with st.expander("Alert packet JSON"):
                st.json(pkt)
        else:
            st.markdown("""
            <div style="background:#161b22;border:1px dashed #21262d;border-radius:10px;
                 padding:3rem 2rem;text-align:center;color:#4f4f4f;margin-top:0.5rem">
              <div style="font-size:1.8rem;margin-bottom:0.6rem">—</div>
              <div style="font-size:0.9rem">Select a tile and click Run Inference</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — ALERT LOG
# ═══════════════════════════════════════════════════════════════
with tab_alerts:
    alerts = load_alerts()
    if not alerts:
        st.markdown("""
        <div style="background:#161b22;border:1px dashed #21262d;border-radius:10px;
             padding:3rem 2rem;text-align:center;color:#4f4f4f;margin-top:1rem">
          <div style="font-size:0.9rem">No alerts yet. Run inference on the Demo tab.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        header = st.columns([2, 1.2, 1, 0.8, 0.8])
        for h, t in zip(header, ["Tile", "Timestamp UTC", "Status", "Coverage", "Severity"]):
            h.markdown(f'<p class="section-label">{t}</p>', unsafe_allow_html=True)
        st.divider()
        for a in reversed(alerts):
            det = a["detection"]; sev = a["severity"]
            ts  = a["timestamp_utc"][:19].replace("T", " ")
            tile = Path(a["source"]["tile_path"]).name
            sc = "#e03131" if a["status"] == "ALERT" else "#4f98a3"
            vc = SEV_FG.get(sev["label"], "#aaa")
            with st.expander(f"{tile[:50]}"):
                r1,r2,r3,r4,r5 = st.columns([2,1.2,1,0.8,0.8])
                r1.caption(tile); r2.caption(ts)
                r3.markdown(f'<span style="color:{sc};font-weight:700;font-size:0.82rem">{a["status"]}</span>', unsafe_allow_html=True)
                r4.caption(f'{det["plume_coverage_pct"]}%')
                r5.markdown(f'<span style="color:{vc};font-weight:700;font-size:0.82rem">{sev["label"]}</span>', unsafe_allow_html=True)
                st.json(a)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODEL CARD (redesigned)
# ═══════════════════════════════════════════════════════════════
with tab_model:

    # ── Row 1: Architecture + Config ──────────────────────────
    col_arch, col_cfg = st.columns([1, 1], gap="large")

    with col_arch:
        st.markdown('<p class="section-label">Architecture</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="arch-wrapper">

          <div class="arch-block">
            <div class="arch-block-title arch-neutral">Input</div>
            <div class="arch-block-body">
              Sentinel-2 tile &nbsp;·&nbsp; bands B11 (SWIR-1) + B12 (SWIR-2)<br>
              Shape &nbsp;<code style="background:#0d1117;padding:1px 6px;border-radius:4px">2 × 512 × 512</code>
              &nbsp;·&nbsp; 20 m/px resolution
            </div>
          </div>

          <div class="arch-arrow">&#8595;</div>

          <div class="arch-block">
            <div class="arch-block-title arch-frozen">Encoder &nbsp;·&nbsp; FROZEN</div>
            <div class="arch-block-body">
              TerraMind-1.0-small (foundation model)<br>
              21,481,728 params &nbsp;·&nbsp; no gradient updates<br>
              Output &nbsp;<code style="background:#0d1117;padding:1px 6px;border-radius:4px">196 × 384</code>&nbsp; patch embeddings
            </div>
          </div>

          <div class="arch-arrow">&#8595;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8595;</div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem">
            <div class="arch-block">
              <div class="arch-block-title arch-train">Seg Decoder &nbsp;·&nbsp; TRAINABLE</div>
              <div class="arch-block-body">
                5× TransposeConv2d<br>+ BatchNorm + ReLU<br>
                Output &nbsp;<code style="background:#0d1117;padding:1px 6px;border-radius:4px">512 × 512</code>&nbsp; binary mask
              </div>
            </div>
            <div class="arch-block">
              <div class="arch-block-title arch-train">Severity Head &nbsp;·&nbsp; TRAINABLE</div>
              <div class="arch-block-body">
                Linear(384 → 128) + ReLU<br>
                Dropout(0.3)<br>
                Linear(128 → 3)&nbsp;
                <span style="color:#6daa45">L</span>&nbsp;
                <span style="color:#ff8c00">M</span>&nbsp;
                <span style="color:#e03131">H</span>
              </div>
            </div>
          </div>

          <div class="arch-arrow">&#8595;</div>

          <div class="arch-block" style="border-color:#01696f22">
            <div class="arch-block-title arch-neutral">Output</div>
            <div class="arch-block-body">
              Binary plume mask &nbsp;+&nbsp; severity label &nbsp;+&nbsp; JSON alert packet<br>
              <span style="color:#4f4f4f">616,388 trainable params &nbsp;·&nbsp; 85 MB checkpoint</span>
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

    with col_cfg:
        st.markdown('<p class="section-label">Training Configuration</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="cfg-grid">
          <div class="cfg-item">
            <div class="cfg-key">Optimizer</div>
            <div class="cfg-val">AdamW</div>
          </div>
          <div class="cfg-item">
            <div class="cfg-key">Learning rate</div>
            <div class="cfg-val">1 × 10<sup style="font-size:0.65rem">-3</sup></div>
          </div>
          <div class="cfg-item">
            <div class="cfg-key">Weight decay</div>
            <div class="cfg-val">1 × 10<sup style="font-size:0.65rem">-4</sup></div>
          </div>
          <div class="cfg-item">
            <div class="cfg-key">Scheduler</div>
            <div class="cfg-val">ReduceLROnPlateau</div>
          </div>
          <div class="cfg-item">
            <div class="cfg-key">Patience</div>
            <div class="cfg-val">5 epochs</div>
          </div>
          <div class="cfg-item">
            <div class="cfg-key">Epochs trained</div>
            <div class="cfg-val">20</div>
          </div>
          <div class="cfg-item">
            <div class="cfg-key">Batch size</div>
            <div class="cfg-val">8</div>
          </div>
          <div class="cfg-item">
            <div class="cfg-key">Seg loss weight</div>
            <div class="cfg-val">0.7</div>
          </div>
          <div class="cfg-item" style="grid-column:span 2">
            <div class="cfg-key">Seg loss</div>
            <div class="cfg-val" style="font-size:0.82rem">0.5 × BCE &nbsp;+&nbsp; 0.5 × DiceLoss</div>
          </div>
          <div class="cfg-item" style="grid-column:span 2">
            <div class="cfg-key">Severity loss</div>
            <div class="cfg-val" style="font-size:0.82rem">CrossEntropy &nbsp;(class-weighted) &nbsp;·&nbsp; weight 0.3</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Known Limitations</p>', unsafe_allow_html=True)
        limitations = [
            ("IoU ceiling 0.33", "noisy MBMP pseudo-labels cap supervision quality; best val IoU 0.3257"),
            ("Severity proxy",   "coverage-% used as emission-rate proxy — not physically grounded"),
            ("No temporal fusion", "single acquisition; surface artifacts cause false positives"),
            ("CRS mismatch",     "centroid output in projected CRS, not WGS84 lat/lon"),
            ("Limited eval",     "validated on 10 geographic locations only"),
        ]
        for title, body in limitations:
            st.markdown(
                f'<div class="lim-card"><b>{title}</b> — {body}</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Row 2: Baselines table + Training charts ───────────────
    col_base, col_chart = st.columns([1, 1.2], gap="large")

    with col_base:
        st.markdown('<p class="section-label">Baseline Comparison</p>', unsafe_allow_html=True)

        # Pre-format numeric columns as strings to avoid jinja2 / Styler dependency
        df_display = pd.DataFrame({
            "Model"      : ["Physics (MBMP)", "TinyUNet (scratch)", "MethanePulse (ours)"],
            "Val IoU"    : ["0.1210", "0.4897", "0.3257 ★"],
            "Val F1"     : ["0.2080", "0.6575", "0.5112 ★"],
            "Precision"  : ["—",      "0.7565", "—"],
            "Recall"     : ["0.2203", "0.5813", "0.5110"],
            "FP Rate"    : ["0.0044", "0.0010", "—"],
            "Params"     : ["0",      "~2 M",   "616 K"],
            "Size (MB)"  : ["—",      "~8",     "85"],
            "Trainable"  : ["None",   "Full",   "Decoder + Head"],
        })
        st.dataframe(df_display, hide_index=True, height=145, width="stretch")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Mini bar chart — IoU comparison
        fig, ax = plt.subplots(figsize=(5.5, 2.4), facecolor="#0d1117")
        models  = ["Physics\n(MBMP)", "TinyUNet\n(scratch)", "MethanePulse\n(ours)"]
        metrics = {
            "Val IoU" : ([0.1210, 0.4897, 0.3257], "#01696f"),
            "Val F1"  : ([0.2080, 0.6575, 0.5112], "#4f98a3"),
            "Recall"  : ([0.2203, 0.5813, 0.5110], "#6daa45"),
        }
        x     = np.arange(len(models))
        width = 0.26
        ax.set_facecolor("#0d1117")
        for i, (label, (vals, color)) in enumerate(metrics.items()):
            bars = ax.bar(x + i * width, vals, width, label=label,
                          color=color, edgecolor="#161b22", linewidth=0.5)
        ax.set_xticks(x + width); ax.set_xticklabels(models, fontsize=8, color="#8b949e")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_visible(False)
        ax.set_ylim(0, 0.80)
        ax.set_ylabel("Score", color="#8b949e", fontsize=8)
        ax.legend(facecolor="#161b22", labelcolor="#8b949e", fontsize=7.5,
                  framealpha=0.8, loc="upper left", ncol=3)
        fig.tight_layout(pad=0.8)
        st.pyplot(fig); plt.close()

    with col_chart:
        st.markdown('<p class="section-label">Training History</p>', unsafe_allow_html=True)
        history = load_history()
        if history:
            epochs   = [h["epoch"]      for h in history]
            tr_loss  = [h["train_loss"] for h in history]
            val_loss = [h["val_loss"]   for h in history]
            iou_vals = [h["iou"]        for h in history]
            f1_vals  = [h.get("f1", None) for h in history]
            has_f1   = any(v is not None for v in f1_vals)
            lr_vals  = [h.get("lr", None) for h in history]
            has_lr   = any(v is not None for v in lr_vals)

            rows = (2 + (1 if has_lr else 0) + (1 if has_f1 else 0))
            fig  = plt.figure(figsize=(6, 1.8 * rows + 1.2), facecolor="#0d1117")
            gs   = gridspec.GridSpec(rows, 1, hspace=0.55, figure=fig)

            def style_ax(ax, ylabel):
                ax.set_facecolor("#0d1117")
                ax.tick_params(colors="#8b949e", labelsize=8)
                ax.spines[:].set_visible(False)
                ax.set_ylabel(ylabel, color="#8b949e", fontsize=8)
                ax.yaxis.set_tick_params(labelcolor="#8b949e")
                ax.xaxis.set_tick_params(labelcolor="#8b949e")
                ax.grid(axis="y", color="#21262d", linewidth=0.6, linestyle="--")

            # Loss
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(epochs, tr_loss,  color="#4f98a3", lw=1.8, label="Train loss")
            ax1.plot(epochs, val_loss, color="#e03131", lw=1.8, ls="--", label="Val loss")
            style_ax(ax1, "Loss")
            ax1.legend(facecolor="#161b22", labelcolor="#8b949e", fontsize=8,
                       framealpha=0.7, loc="upper right", ncol=2)

            # IoU
            ax2 = fig.add_subplot(gs[1])
            ax2.fill_between(epochs, iou_vals, alpha=0.12, color="#6daa45")
            ax2.plot(epochs, iou_vals, color="#6daa45", lw=1.8)
            best_ep  = epochs[iou_vals.index(max(iou_vals))]
            best_iou = max(iou_vals)
            ax2.axvline(best_ep, color="#6daa45", ls=":", alpha=0.5, lw=1)
            ax2.scatter([best_ep], [best_iou], color="#6daa45", s=40, zorder=5)
            ax2.annotate(
                f"Best {best_iou:.4f}",
                xy=(best_ep, best_iou),
                xytext=(best_ep + 0.4, best_iou - 0.007),
                color="#6daa45", fontsize=8
            )
            style_ax(ax2, "Val IoU")

            # F1 (optional — present in new training runs)
            next_gs = 2
            if has_f1:
                ax_f1 = fig.add_subplot(gs[next_gs])
                ax_f1.fill_between(epochs, f1_vals, alpha=0.10, color="#4f98a3")
                ax_f1.plot(epochs, f1_vals, color="#4f98a3", lw=1.8)
                best_f1_ep  = epochs[f1_vals.index(max(f1_vals))]
                best_f1_val = max(f1_vals)
                ax_f1.axvline(best_f1_ep, color="#4f98a3", ls=":", alpha=0.5, lw=1)
                ax_f1.scatter([best_f1_ep], [best_f1_val], color="#4f98a3", s=40, zorder=5)
                ax_f1.annotate(
                    f"Best {best_f1_val:.4f}",
                    xy=(best_f1_ep, best_f1_val),
                    xytext=(best_f1_ep + 0.4, best_f1_val - 0.006),
                    color="#4f98a3", fontsize=8
                )
                style_ax(ax_f1, "Val F1")
                next_gs = 3

            # LR (optional)
            if has_lr:
                ax3 = fig.add_subplot(gs[next_gs])
                ax3.plot(epochs, lr_vals, color="#8b949e", lw=1.4)
                style_ax(ax3, "LR")
                ax3.set_xlabel("Epoch", color="#8b949e", fontsize=8)
            elif has_f1:
                ax_f1.set_xlabel("Epoch", color="#8b949e", fontsize=8)
            else:
                ax2.set_xlabel("Epoch", color="#8b949e", fontsize=8)

            fig.patch.set_facecolor("#0d1117")
            st.pyplot(fig); plt.close()
        else:
            st.info("No training history found in docs/")