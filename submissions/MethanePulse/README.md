# MethanePulse: Automated Methane Plume Detection from Sentinel-2 SWIR

**Team:** MethanePulse-Leo
**Track:** AI/ML
**Hackathon:** TM2Space x Aeon418

---

## 1. What problem are we solving?

Methane is 80x more powerful than CO2 over a period of 20 years. But the issue isn’t that we don’t have satellites that can observe this gas. The issue is that no one is monitoring quickly enough.

The emission source from industries is continuous – be it from an oil rig or a landfill. Currently, environmental agencies, ESG compliance departments, or energy firms under regulatory pressures have to hire analysts who manually review the tiles from satellites. This process is inefficient, costly, and is structurally destined to fail because of missed alerts that happen during this time window. An event that happens between 9 am and noon will remain unrecorded.

MethanePulse solves this problem of monitoring entirely by removing humans from the loop. Provided the raw data from a Sentinel-2 image, it produces an output plume mask, along with a severity label (LOW, MEDIUM, HIGH), and an alert packet in a single pass. Humans only get involved in case of a HIGH label being generated.

---

## 2. What did we build?

The core of MethanePulse is a dual-head model fine-tuned on top of TerraMind-1.0-small, the IBM/ESA geospatial foundation model, with the encoder kept fully frozen throughout training.

Input is a 2-channel SWIR stack using Sentinel-2 bands B11 and B12 at 20m resolution, tiled to 512x512 patches. The frozen encoder converts each tile into patch embeddings that already encode rich Earth observation priors. A 5-layer transposed-conv decoder then upsamples those embeddings back to a full 512x512 binary plume mask. Simultaneously, a linear severity head reads the CLS token and classifies the tile into one of three severity bins.

The key design choice here was freezing the foundation model entirely. This kept trainable parameters at 616,388, which is a fraction of what a full fine-tune would require, and forced the heads to work with features that already understand spectral and spatial structure at a planetary scale.

Training ran for 15 epochs using AdamW (lr=1e-3, wd=1e-4) with a combined segmentation and severity loss weighted at 0.8 and 0.2 respectively. Learning rate scheduling used ReduceLROnPlateau with patience=3. Binary plume pseudo-labels came from MBMP band-ratio thresholding. Severity thresholds were derived from the P33 and P66 percentiles of plume coverage area in the training split.

---

## 3. How did we measure it?

IoU on held-out validation tiles was the primary metric. 3-class severity accuracy was secondary.

| Model | Val IoU | Severity Accuracy | Notes |
|---|---|---|---|
| Physics baseline (MBMP) | 0.2037 | n/a | Band-ratio threshold, no ML |
| TinyUNet (scratch) | 0.2779 | 0.61 | 4-layer UNet, same data |
| MethanePulse (ours) | **0.3298** | **0.68** | Frozen FM + trained heads |

MethanePulse achieves a +62% IoU improvement over the physics-only baseline and +18.7% over a UNet trained from scratch on the same data. The result matters specifically because the foundation model was never trained on methane detection. The gain confirms that frozen geospatial features generalise to a novel spectral detection task without any encoder updates.

---

## 4. What is the orbital-compute story?

The complete checkpoint size is 85 MB. It can fit onto solid-state onboard memory of a CubeSat-level spacecraft without any need for compression.

There being no encoder training once frozen, there is no retraining process in orbit. A forward pass using Apple MPS (there being no AI accelerator) takes about 8 seconds. Running the same operation on an NVIDIA Jetson AGX Orin, which is the hardware used by ESA’s Phi-Sat missions, takes less than 10 minutes to complete; hence, the processed data and alert packages can be downlinked during the pass itself.

This dual-head approach is carefully thought through. Only one inference step is required in order to produce both the segmentation mask and the classification output. There are no second models to be called, no additional post-processing steps to perform triage, nor any added computations costs.

INT8 quantization (yet to be performed) will reduce the size of the trainable model head to about 2 MB, making deployment onto Myriad X VPU or comparable radiation-hardened embedded chipsets possible. Inference is fully deterministic since the encoder is frozen, and there is no mutation of any weight values, meeting the hard reproducibility requirements of space-qualified flight software.

---

## 5. What does not work yet?

**The IoU ceiling has not been reached.** 0.33 is where we start, not where the model tops out. The current supervision signal comes from MBMP pseudo-labels, which are noisy by construction. Replacing them with human-annotated masks would directly improve the segmentation head without any architectural changes.

**Severity labels are coverage proxies, not emission estimates.** Classifying a plume as HIGH because it covers a large tile area is not the same as knowing it emits X kg/hr. Connecting severity to calibrated emission rates requires radiative transfer modelling and ground truth that was not available in this training set.

**There is no temporal context.** TerraMind natively supports multi-date inputs. Diffing B12 across two acquisitions of the same site would catch transient plumes and filter out false positives caused by surface reflectance features that look spectrally similar to methane absorption. This is the single highest-value improvement on the roadmap.

**Alert coordinates are not in WGS84.** Plume centroids in the JSON output are currently reported in the tile's projected CRS. A one-line rasterio transform converts them to standard lat/lon. It is not applied yet.

**Out-of-distribution performance is unknown.** Training and validation used one dataset from one set of geographic locations and acquisition conditions. How the model behaves on Arctic tundra, tropical wetlands, or dry-season desert tiles has not been tested.

---

## Running inference

```bash
pip install -r requirements.txt

python infer.py \
  --input  sample_input/0___0_S2B_20240326T165849_R1000_nonorm_bands20m.tiff \
  --checkpoint terramind_methane_best.pt \
  --output results/
```

The run produces two files: `results/<tile>_mask.tiff` containing the binary plume mask, and `results/<tile>_alert.json` containing the severity label, centroid coordinates, and coverage statistics.

Expected runtime is roughly 8 seconds on Apple MPS and 5 seconds on a CUDA GPU.

---

## Checkpoint and dataset

`terramind_methane_best.pt` is included in the submission (85 MB). It represents the best validation checkpoint, reaching IoU 0.3298 at epoch 15.

The training dataset is GeoCH4PlumeNet: 1,000 Sentinel-2 L1C tiles at 512x512 with all 13 bands, covering 147 simulated plumes across 10 global oil and gas sites, split 70/10/20 for train, validation, and test. It includes binary plume masks and emission rate labels and was funded by OHB Digital Connect GmbH and ESA. The dataset is not included in the submission due to size (over 14 GB) and is available at https://zenodo.org/records/16813369
