from __future__ import annotations

import argparse
import sys
from pathlib import Path

from repo.train.service_http import JsonApiServer


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL = REPO_ROOT / "Local_Emphasis"
if str(LOCAL) not in sys.path:
    sys.path.insert(0, str(LOCAL))

from local_emphasis_reward import LocalEmphasisReward  # noqa: E402


app = JsonApiServer()
REWARD = LocalEmphasisReward(preset="paper_baseline")


def _load_wav(path: str):
    import soundfile as sf
    import torch

    data, sr = sf.read(path, always_2d=True)  # [T, C]
    wav = torch.from_numpy(data.T).float()    # [C, T]
    return wav.mean(dim=0), int(sr)


@app.route("/score")
def score(payload: dict) -> dict:
    wav, sr = _load_wav(payload["audio_path"])
    boundaries = payload.get("word_boundaries") or []
    target_indices = payload.get("target_indices") or []

    if not boundaries or not target_indices:
        return {"ok": True, "total": 0.0}

    # Average reward over all target indices
    total = 0.0
    for target_idx in target_indices:
        if target_idx < 0 or target_idx >= len(boundaries):
            continue
        r = REWARD.compute(wav, sr, [tuple(b) for b in boundaries], int(target_idx))
        total += float(r.get("total", 0.0))
    total = total / max(len(target_indices), 1)

    return {"ok": True, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local emphasis reward HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    app.serve(args.host, args.port)


if __name__ == "__main__":
    main()

