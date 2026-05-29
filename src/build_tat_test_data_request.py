import json
import os
import time
import threading
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


API_URL = "http://tts001.bronci.com.tw:8802/run/predict"
RATE_LIMIT_MSG = "請求過於頻繁"

_request_count = 0


def translate_hanji_to_zh(text: str) -> str:
    global _request_count
    _request_count += 1
    if _request_count % 11 == 0:
        time.sleep(5)

    data = {
        "data": ["", text, "taigi_tw_zh"],
        "event_data": None,
        "fn_index": 0,
        "session_hash": "build_tat",
    }
    wait = 10
    while True:
        try:
            r = requests.post(API_URL, json=data, headers={"Content-Type": "application/json"}, timeout=15)
            if r.status_code == 200:
                result = r.json()
                if "data" in result and result["data"]:
                    text_result = result["data"][0]
                    if RATE_LIMIT_MSG in text_result:
                        print(f"  [rate limit] 等待 {wait} 秒後重試...", flush=True)
                        time.sleep(wait)
                        wait = min(wait * 2, 60)
                        continue
                    return text_result
        except Exception as e:
            print(f"[ERROR] ({text[:20]}...): {e}", flush=True)
            time.sleep(5)
        wait = 10


def load_all_json(json_dir: Path) -> list[dict]:
    records = []
    for speaker_dir in sorted(json_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for json_file in sorted(speaker_dir.glob("*.json")):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            speaker = data.get("發音人", speaker_dir.name)
            card_no = data.get("提示卡編號", "")
            sent_no = data.get("句編號", "")
            hanji = data.get("漢羅台文", "")
            record_id = f"{speaker}:{card_no}-{sent_no}"
            records.append({"id": record_id, "hanji": hanji})
    return records


def process_record(record: dict) -> dict:
    translated = translate_hanji_to_zh(record["hanji"])
    return {"ID": record["id"], "Text": translated}


def main():
    json_dir = Path("/srv/RL_project/TAT-Vol1/TAT-Vol1-test/json")
    output_path = json_dir / "tat_test_task3.jsonl"

    print("載入 JSON 檔案...", flush=True)
    records = load_all_json(json_dir)
    print(f"共 {len(records)} 句，開始翻譯（並發 8 threads）...", flush=True)

    results = [None] * len(records)
    done = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_idx = {executor.submit(process_record, rec): i for i, rec in enumerate(records)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if done % 100 == 0:
                print(f"  進度：{done}/{len(records)}", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"完成！輸出至 {output_path}（共 {len(results)} 筆）", flush=True)


if __name__ == "__main__":
    main()
