from __future__ import annotations

import argparse
import sys
from pathlib import Path

from repo.train.service_http import JsonApiServer


REPO_ROOT = Path(__file__).resolve().parents[2]
EMOTION2VEC = REPO_ROOT / "emotion2vec"
if str(EMOTION2VEC) not in sys.path:
    sys.path.insert(0, str(EMOTION2VEC))

from test import normalize_emotion, predict_emotion  # noqa: E402


app = JsonApiServer()
MODEL = None
OUTPUT_DIR = None


@app.route("/classify")
def classify(payload: dict) -> dict:
    if MODEL is None:
        raise RuntimeError("emotion2vec model is not loaded")
    pred, top_score, labels, scores = predict_emotion(MODEL, Path(payload["audio_path"]), OUTPUT_DIR)
    pred_norm = normalize_emotion(pred)
    target = normalize_emotion(payload.get("target_emotion"))
    reward = 5.0 if pred_norm == target else -1.0
    return {
        "ok": True,
        "pred_emotion": pred_norm,
        "target_emotion": target,
        "top_score": float(top_score),
        "reward": reward,
        "labels": labels,
        "scores": scores,
    }


def main() -> None:
    global MODEL, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Emotion2Vec reward HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-dir", default=str(REPO_ROOT.parent / "models" / "emotion2vec_plus_large"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "emotion2vec_service"))
    args = parser.parse_args()
    from funasr import AutoModel

    OUTPUT_DIR = Path(args.output_dir)
    MODEL = AutoModel(model=str(Path(args.model_dir).expanduser().resolve()), hub="hf", disable_update=True)
    app.serve(args.host, args.port)


if __name__ == "__main__":
    main()

