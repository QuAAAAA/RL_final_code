import torch
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL = "google/gemma-3-12b-it"
ADAPTER_PATH = "/home/black0000/workspace/CosyVoice/pretrained_mt_models/checkpoint-2456"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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
    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()


# 測試句子（台羅數字調）
test_cases = [
    "sua3-loh8-lai5 khuann3 sin1-tik4-tshi7 bin5-a2-tsai3 it4 ho7 e5 thinn1-khi3",
    "un1-too7 ji7-tsap3-sann1 kau3 ji7-tsap3-tshit4 too7 lo7-hi2 ki1-lu7t4 ji7-tsap3 %",
    "lai5-pin1 go7-pah4-go7-tsap3-kiu2 ho7 tshiann2 lai5 tsap3-sann1 ho7 kui7-tai5 pan7-li2",
]

print("\n--- 測試結果 ---", flush=True)
for src in test_cases:
    result = tw2zh(src)
    print(f"輸入：{src}", flush=True)
    print(f"輸出：{result}", flush=True)
    print(flush=True)
