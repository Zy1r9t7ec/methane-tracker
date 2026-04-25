import json
import uuid
import torch
import numpy as np
import rasterio
from datetime import datetime, timezone
from pathlib import Path

SEVERITY_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
SEVERITY_COLOR  = {0: "#f5c518", 1: "#ff8c00", 2: "#e03131"}

def generate_alert_packet(tiff_path, model, device, derive_severity_fn,
                           pair_image_label_fn, label_dir,
                           confidence_threshold=0.5):
    tiff_path = Path(tiff_path)
    with rasterio.open(tiff_path) as src:
        img       = src.read().astype(np.float32)
        crs       = str(src.crs)
        bounds    = src.bounds
        transform = src.transform

    eps = 1e-6
    b11, b12 = img[11], img[12]

    def norm(x):
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / (hi - lo + eps), 0, 1).astype(np.float32)

    x_tensor = torch.tensor(
        np.stack([norm(b11), norm(b12)], axis=0),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        seg_logits, sev_logits = model(x_tensor)

    prob_map  = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    pred_mask = (prob_map > confidence_threshold).astype(np.uint8)
    sev_class = int(sev_logits.argmax(dim=1).item())
    sev_conf  = float(torch.softmax(sev_logits, dim=1).max().item())

    plume_px  = int(pred_mask.sum())
    total_px  = pred_mask.size
    plume_pct = round(100 * plume_px / total_px, 4)
    mean_prob = round(float(prob_map[pred_mask == 1].mean())
                      if plume_px > 0 else 0.0, 4)

    if plume_px > 0:
        ys, xs       = np.where(pred_mask == 1)
        centroid_px  = [int(xs.mean()), int(ys.mean())]
        cx_geo       = transform.c + centroid_px[0] * transform.a
        cy_geo       = transform.f + centroid_px[1] * transform.e
        centroid_geo = [round(cx_geo, 6), round(cy_geo, 6)]
    else:
        centroid_px  = None
        centroid_geo = None

    gt_iou = None
    gt_severity = None
    try:
        lbl_path = pair_image_label_fn(tiff_path, label_dir)
        with rasterio.open(lbl_path) as src:
            gt_mask = (src.read(1) > 0.5).astype(np.uint8)
        tp = int(((pred_mask == 1) & (gt_mask == 1)).sum())
        fp = int(((pred_mask == 1) & (gt_mask == 0)).sum())
        fn = int(((pred_mask == 0) & (gt_mask == 1)).sum())
        gt_iou      = round(tp / (tp + fp + fn + 1e-6), 4)
        gt_severity = int(derive_severity_fn(gt_mask.astype(float)))
    except Exception:
        pass

    return {
        "alert_id"      : str(uuid.uuid4()),
        "schema_version": "1.0",
        "timestamp_utc" : datetime.now(timezone.utc).isoformat(),
        "source": {
            "tile_path" : str(tiff_path),
            "sensor"    : "Sentinel-2",
            "bands_used": ["B11 (SWIR_1)", "B12 (SWIR_2)"],
            "crs"       : crs,
            "bounds"    : {
                "left"  : round(bounds.left,   6),
                "bottom": round(bounds.bottom, 6),
                "right" : round(bounds.right,  6),
                "top"   : round(bounds.top,    6)
            }
        },
        "detection": {
            "plume_detected"      : bool(plume_px > 0),
            "confidence_threshold": confidence_threshold,
            "plume_pixel_count"   : plume_px,
            "plume_coverage_pct"  : plume_pct,
            "mean_plume_prob"     : mean_prob,
            "centroid_pixel"      : centroid_px,
            "centroid_geographic" : centroid_geo
        },
        "severity": {
            "class_id"  : sev_class,
            "label"     : SEVERITY_LABELS[sev_class],
            "confidence": round(sev_conf, 4),
            "color_hex" : SEVERITY_COLOR[sev_class]
        },
        "model": {
            "name"            : "TerraMindMethanePulse",
            "encoder"         : "TerraMind-1.0-small (frozen)",
            "trainable_params": 616388,
            "best_val_iou"    : 0.3298
        },
        "ground_truth": {
            "iou"          : gt_iou,
            "severity_class": gt_severity
        } if gt_iou is not None else None,
        "status": "ALERT" if plume_px > 0 else "CLEAR"
    }

def save_alert(packet, out_dir="outputs/alerts"):
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{packet['alert_id']}.json"
    with open(out_path, "w") as f:
        json.dump(packet, f, indent=2)
    print(f"Alert saved → {out_path}")
    return out_path
