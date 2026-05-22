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
REWARD = LocalEmphasisReward()


def _load_wav(path: str):
    import soundfile as sf
    import torch

    data, sr = sf.read(path, always_2d=True)  # [T, C]
    wav = torch.from_numpy(data.T).float()    # [C, T]
    return wav.mean(dim=0), int(sr)


@app.route("/score")
def score(payload: dict) -> dict:
    wav, sr = _load_wav(payload["audio_path"])
    boundaries = payload.get("word_boundaries")
    targets = payload.get("emphasis_targets") or []
    if not boundaries:
        # Placeholder until NeMo alignment is wired in: equal-width pseudo boundaries.
        words = [str(item.get("word")) for item in targets if item.get("word")]
        dur = wav.numel() / float(sr)
        step = dur / max(len(words), 1)
        boundaries = [(word, idx * step, (idx + 1) * step) for idx, word in enumerate(words)]
    if not boundaries:
        return {"ok": True, "total": 0.0}
    result = REWARD.compute(wav, sr, boundaries, int(payload.get("target_idx", 0)))
    return {"ok": True, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local emphasis reward HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    app.serve(args.host, args.port)


if __name__ == "__main__":
    main()

