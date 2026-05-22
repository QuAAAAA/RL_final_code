from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from repo.train.service_http import JsonApiServer


REPO_ROOT = Path(__file__).resolve().parents[2]
W2V2_DIR = REPO_ROOT / "w2v2-how-to"
INFERENCE_PATH = W2V2_DIR / "inference.py"

spec = importlib.util.spec_from_file_location("w2v2_vad_inference", INFERENCE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load VAD inference module: {INFERENCE_PATH}")
w2v2_vad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w2v2_vad)


app = JsonApiServer()
INTERFACE = None
MODEL_ROOT = W2V2_DIR / "model"
CACHE_ROOT = W2V2_DIR / "cache"


@app.route("/vad")
def vad(payload: dict) -> dict:
    if INTERFACE is None:
        raise RuntimeError("VAD interface is not loaded")
    result = w2v2_vad.predict_vad(payload["audio_path"], interface=INTERFACE)
    return {"ok": True, **result}


@app.route("/va")
def va(payload: dict) -> dict:
    if INTERFACE is None:
        raise RuntimeError("VAD interface is not loaded")
    result = w2v2_vad.predict_va(payload["audio_path"], interface=INTERFACE)
    return {"ok": True, **result}


def main() -> None:
    global INTERFACE, MODEL_ROOT, CACHE_ROOT
    parser = argparse.ArgumentParser(description="Wav2Vec2 VAD/VA HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--cache-root", default=str(CACHE_ROOT))
    args = parser.parse_args()

    MODEL_ROOT = Path(args.model_root).expanduser().resolve()
    CACHE_ROOT = Path(args.cache_root).expanduser().resolve()
    model = w2v2_vad.setup_model(model_root=MODEL_ROOT, cache_root=CACHE_ROOT)
    INTERFACE = w2v2_vad.setup_interface(model=model, model_root=MODEL_ROOT, cache_root=CACHE_ROOT)
    app.serve(args.host, args.port)


if __name__ == "__main__":
    main()
