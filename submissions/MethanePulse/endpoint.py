"""
MethanePulse — Flask inference endpoint
POST /predict   multipart/form-data
  file      : .tiff tile
  threshold : float (optional, default 0.5)
"""

from flask import Flask, request, jsonify
from pathlib import Path
import tempfile, sys, os

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from infer import run_inference

app = Flask(__name__)

CKPT    = BASE_DIR / "terramind_methane_best.pt"
RESULTS = BASE_DIR / "results"


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate file ─────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "missing 'file' field in multipart/form-data"}), 400

    uploaded  = request.files["file"]
    threshold = float(request.form.get("threshold", 0.5))

    if not CKPT.exists():
        return jsonify({"error": f"Checkpoint not found: {CKPT}"}), 500

    # ── Save upload to a temp file ────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
    try:
        uploaded.save(tmp.name)
        tmp.close()

        # ── Run inference (returns the full alert dict) ───────────
        alert = run_inference(
            tiff_path             = tmp.name,
            checkpoint_path       = str(CKPT),
            output_dir            = str(RESULTS),
            confidence_threshold  = threshold,
        )

        return jsonify(alert), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)