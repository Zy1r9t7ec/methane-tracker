# MethanePulse — Methane Plume Detection from Sentinel-2 SWIR

**Team:** MethanePulse-Leo | **Track:** AI/ML | **Hackathon:** TM2Space × Aeon418

---

## 1. What problem are we solving?

Methane is 80× more potent than CO₂ over a 20-year horizon, and industrial point sources —
oil wells, pipelines, landfills — emit undetected plumes daily. The customer is any operator
running a methane monitoring service (environmental agencies, ESG compliance teams, energy
companies under regulatory pressure): they currently pay analysts to manually scan satellite
imagery, which costs time and misses events between review cycles. MethanePulse automates the
detection pipeline — given a raw Sentinel-2 tile, it outputs a binary plume mask, a severity
triage label (LOW / MEDIUM / HIGH), and a structured JSON alert packet, all in one forward pass.
Zero analyst time until a HIGH alert fires.

---

## 2. What did we build?

We fine-tuned a dual-head model on top of the frozen TerraMind-1.0-small geospatial foundation
model (IBM/ESA). Input is 2-channel SWIR (B11 + B12 at 20m resolution, 512×512 tiles). The
frozen encoder produces patch embeddings; a 5-layer transposed-conv decoder upsamples to a
512×512 binary plume mask, while a linear severity head classifies coverage intensity from the
CLS token. Only 616,388 parameters are trainable. Training used AdamW (lr=1e-3, wd=1e-4),
combined seg+severity loss (SEG_WEIGHT=0.8, SEV_WEIGHT=0.2), ReduceLROnPlateau (patience=3),
for 15 epochs on the hackathon-provided Sentinel-2 dataset. Labels were derived from MBMP
thresholding; severity thresholds set at P33/P66 of training plume coverage distribution.

---

## 3. How did we measure it?

Primary metric: IoU on held-out validation tiles. Secondary: 3-class severity accuracy.

| Model                        | Val IoU | Sev Accuracy | Notes                        |
|------------------------------|---------|--------------|------------------------------|
| Physics baseline (MBMP)      | 0.2037  | —            | Band-ratio threshold, no ML  |
| TinyUNet (trained from scratch) | 0.2779 | 0.61        | 4-layer UNet, same data      |
| **TerraMindMethanePulse (ours)** | **0.3298** | **0.68** | Frozen FM + trained heads |

TerraMind improves IoU by +62% over the physics baseline and +18.7% over the from-scratch UNet,
confirming that frozen geospatial foundation model features transfer effectively to plume
segmentation even with noisy pseudo-labels.

---

## 4. The orbital-compute story

The checkpoint is **85 MB** — small enough for solid-state onboard storage on a CubeSat-class
platform. The encoder is permanently frozen, so there is no retraining in orbit. A single
forward pass takes ~8 seconds on Apple MPS (no dedicated AI chip); on an NVIDIA Jetson AGX
Orin — the class of hardware used in ESA's Φ-Sat missions — this would run at real-time rates
well within a 10-minute ground-station window. The dual-head architecture means one inference
call produces both the spatial mask and the severity triage, so downstream alerting needs
zero additional compute. With INT8 quantisation (not yet applied), the head weights compress
to ~2MB, making deployment on a Myriad X VPU or equivalent radiation-tolerant edge AI chip
entirely feasible. The frozen FM means orbital inference is deterministic and needs no
weight update — a hard requirement for space-qualified software.

---

## 5. What doesn't work yet

- **IoU of 0.33 is a starting point, not a ceiling.** MBMP pseudo-labels are noisy;
  human-annotated masks would substantially improve supervision quality.
- **Severity is a proxy, not physics.** Coverage-% thresholds approximate emission rate
  but are not calibrated to kg/hr. Real severity needs radiative transfer ground truth.
- **No temporal fusion.** TerraMind supports multi-date inputs — diffing B12 across two
  acquisitions would cut false positives from surface reflectance artifacts significantly.
- **CRS not normalised.** Alert centroid coordinates are in the tile's projected CRS, not
  WGS84 lat/lon. A one-line `rasterio` transform fixes this but isn't in the current output.
- **Single-scene generalisation unknown.** Model was trained and validated on one dataset;
  performance on other regions, seasons, or sensor configurations is untested.

---

## Running inference

```bash
# Install dependencies
pip install -r requirements.txt

# Run on sample tile
python infer.py \
  --input  sample_input/0___0_S2B_20240326T165849_R1000_nonorm_bands20m.tiff \
  --checkpoint terramind_methane_best.pt \
  --output results/
```

Outputs: `results/<tile>_mask.tiff` (binary plume mask) + `results/<tile>_alert.json`
Expected runtime: ~8s on Apple MPS / ~5s on CUDA GPU.

---

## Checkpoint & dataset

- `terramind_methane_best.pt` — **85 MB**, included. Best val IoU: **0.3298** @ epoch 15.
- Training data: **GeoCH4PlumeNet** — 1,000 Sentinel-2 L1C tiles (512×512, all 13 bands),
  147 simulated plumes from 10 global oil & gas locations, 70/10/20 train/val/test split.
  Binary plume masks + emission rate labels, funded by OHB Digital Connect GmbH & ESA.
  **Not included in submission (>14 GB).**
  Dataset: [https://zenodo.org/records/16813369](https://zenodo.org/records/16813369)
