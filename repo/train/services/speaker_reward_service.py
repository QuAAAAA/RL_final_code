from __future__ import annotations

import argparse
import sys
from pathlib import Path

from repo.train.service_http import JsonApiServer


REPO_ROOT = Path(__file__).resolve().parents[2]
CALCULATE_SS = REPO_ROOT / "calculate_ss"
if str(CALCULATE_SS) not in sys.path:
    sys.path.insert(0, str(CALCULATE_SS))

from cal_ss import speaker_similarity  # noqa: E402


app = JsonApiServer()
MODEL_SOURCE = None
SAVEDIR = None
DEVICE = "cpu"


@app.route("/similarity")
def similarity(payload: dict) -> dict:
    result = speaker_similarity(
        source_audio=payload["source_audio"],
        target_audio=payload["target_audio"],
        model_source=MODEL_SOURCE,
        savedir=SAVEDIR,
        device=DEVICE,
    )
    return {"ok": True, **result}


def main() -> None:
    global MODEL_SOURCE, SAVEDIR, DEVICE
    parser = argparse.ArgumentParser(description="Speaker similarity HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--savedir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    MODEL_SOURCE = args.model
    SAVEDIR = args.savedir
    DEVICE = args.device
    app.serve(args.host, args.port)


if __name__ == "__main__":
    main()

