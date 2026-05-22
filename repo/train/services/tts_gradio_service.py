from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from repo.train.service_http import JsonApiServer


app = JsonApiServer()
CLIENT = None
ARGS = None


@app.route("/synthesize")
def synthesize(payload: dict) -> dict:
    if CLIENT is None or ARGS is None:
        raise RuntimeError("TTS client is not initialized")
    from gradio_client import handle_file

    out_dir = Path(payload.get("output_dir") or ARGS.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = str(payload.get("id") or "utt").replace("/", "_").replace(":", "_")
    out_path = out_dir / f"{uid}.wav"

    prompt_audio = payload.get("prompt_audio") or ARGS.prompt_audio
    generated = CLIENT.predict(
        tts_text=payload["text"],
        mode_checkbox_group=ARGS.mode_checkbox_group,
        prompt_text=payload.get("target_emotion") or "",
        prompt_wav_upload=handle_file(str(Path(prompt_audio).expanduser().resolve())) if prompt_audio else None,
        prompt_wav_record=None,
        seed=int(payload.get("seed") or ARGS.seed),
        speed=ARGS.speed,
        enable_translation=False,
        api_name="/generate",
    )
    shutil.copy(generated, out_path)
    return {"ok": True, "wav_path": str(out_path)}


def main() -> None:
    global CLIENT, ARGS
    parser = argparse.ArgumentParser(description="Gradio TTS proxy HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--server-url", default="https://140.113.30.139:5003/")
    parser.add_argument("--model-name", default="pretrained_For_Selection/台語模型")
    parser.add_argument("--ssl-verify", action="store_true")
    parser.add_argument("--mode-checkbox-group", default="自然語言控制")
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--output-dir", default="repo/train/runs/tts_service/wav")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0)
    ARGS = parser.parse_args()
    from gradio_client import Client

    CLIENT = Client(ARGS.server_url, ssl_verify=ARGS.ssl_verify)
    CLIENT.predict(ARGS.model_name, api_name="/change_model")
    app.serve(ARGS.host, ARGS.port)


if __name__ == "__main__":
    main()
