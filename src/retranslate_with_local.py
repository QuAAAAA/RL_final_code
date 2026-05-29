import json
import re
import torch
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-3-12b-it"
ADAPTER_PATH = "/home/black0000/workspace/CosyVoice/pretrained_mt_models/checkpoint-2456"
INPUT_JSONL = "/srv/RL_project/TAT-Vol1/TAT-Vol1-test/json/tat_test_task3_request.jsonl"
OUTPUT_JSONL = "/srv/RL_project/TAT-Vol1/TAT-Vol1-test/json/tat_test_task3.jsonl"
JSON_BASE = Path("/srv/RL_project/TAT-Vol1/TAT-Vol1-test/json")


def has_latin(text: str) -> bool:
    return bool(re.search(r'[a-zA-Zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜâêîôûǎěǐǒǔ]', text))


def build_index() -> dict:
    idx = {}
    for d in JSON_BASE.iterdir():
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            data = json.load(open(f, encoding="utf-8"))
            speaker = data.get("發音人", "")
            card = data.get("提示卡編號", "")
            sent = data.get("句編號", "")
            key = f"{speaker}:{card}-{sent}"
            idx[key] = data.get("台羅數字調", "")
    return idx


print("建立索引...", flush=True)
idx = build_index()

print("載入模型...", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("模型載入完成", flush=True)


def tw2zh(src_text: str) -> str:
    conversation = [
        {"role": "system", "content": "You are a 'Chinese' and 'Taiwanese Romanization with tone numbers' translation assistant."},
        {"role": "user", "content": f"Translate the following 'Taiwanese Romanization with tone numbers' into 'Chinese':\n{src_text}"},
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", max_length=1172, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    input_len = inputs.input_ids.shape[1]
    result = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
    return result.replace(" ", "")


# 讀取輸入，判斷哪些需要重跑
print("讀取輸入檔案...", flush=True)
entries = []
with open(INPUT_JSONL, encoding="utf-8") as f:
    for line in f:
        entries.append(json.loads(line))

need_retranslate = [(i, e) for i, e in enumerate(entries) if has_latin(e["Text"])]
print(f"共 {len(need_retranslate)} 筆需要重新翻譯", flush=True)

# 重翻
for done, (i, entry) in enumerate(need_retranslate):
    record_id = entry["ID"]
    tailo = idx.get(record_id, "")
    if not tailo:
        print(f"  [SKIP] 找不到台羅數字調：{record_id}", flush=True)
        continue
    new_text = tw2zh(tailo)
    entries[i]["Text"] = new_text
    if (done + 1) % 50 == 0:
        print(f"  進度：{done+1}/{len(need_retranslate)}", flush=True)

# 寫出
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"完成！輸出至 {OUTPUT_JSONL}（共 {len(entries)} 筆）", flush=True)
