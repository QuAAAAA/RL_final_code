from __future__ import annotations

from repo.train.components import control_to_json
from repo.train.dataset import build_gold_control
from repo.train.schemas import TrainExample
from repo.train.service_http import JsonApiServer, server_args


app = JsonApiServer()


@app.route("/predict")
def predict(payload: dict) -> dict:
    example = TrainExample(
        uid=str(payload.get("id") or ""),
        text=str(payload.get("text") or ""),
        tagged_text=str(payload.get("tagged_text") or payload.get("text") or ""),
        quadruplets=list(payload.get("quadruplets") or []),
    )
    return {"ok": True, "control": control_to_json(build_gold_control(example))}


def main() -> None:
    args = server_args("ASQP/VA control service")
    app.serve(args.host, args.port)


if __name__ == "__main__":
    main()

