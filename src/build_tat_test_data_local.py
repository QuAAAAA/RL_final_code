# CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python eval_twpy_ex_260317.py
# CUDA_VISIBLE_DEVICES=0 python eval_twpy_ex_260317.py
from datasets import Dataset, concatenate_datasets
import torch
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
max_length = 1172 #512
base_model_name = "google/gemma-3-12b-it"
model_name = "models/gemma3_12b_zh_twpy_split_5e-4/checkpoint-2456"
flag=True #False #中翻台為True，反之為False
mode="zh2tw" #flag=False，mode 設為 "tw2zh"
print(model_name, base_model_name)
print(flag, mode)

from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
from transformers import AutoTokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config, #VRAM 不足才需要
    trust_remote_code=True,
    device_map="auto",
    # dtype=torch.bfloat16, 
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, model_name)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    padding_side="left",
)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = model.eval()

def format_instruction(src_text, flag=True, return_flag=False, add_g_p=True):
    if flag==True:
        conversation = [
            {"role": "system", "content": "You are a 'Chinese' and 'Taiwanese Romanization with tone numbers' translation assistant."},
            {"role": "user", "content": f"Translate the following 'Chinese' into 'Taiwanese Romanization with tone numbers':\n{src_text}"},
        ]
    else:
        conversation = [
            {"role": "system", "content": "You are a 'Chinese' and 'Taiwanese Romanization with tone numbers' translation assistant."},
            {"role": "user", "content": f"Translate the following 'Taiwanese Romanization with tone numbers' into 'Chinese':\n{src_text}"},
        ]
    if return_flag: return conversation
    else: return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_g_p)

import torch
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import datetime
import math
df = pd.read_excel('0303待轉台羅數字調拼音.xlsx')
src_lines=df['站名'].tolist()
data=[{"chinese": src} for src in src_lines]
dataset = Dataset.from_list(data)
test_dataset = dataset.shuffle(seed=42)
print(f"資料集大小: {len(test_dataset['chinese'])}")

def infer(model, tokenizer, test_dataset):
    predictions = []
    predictions2 = []
    for src in tqdm(test_dataset["chinese"], total=len(test_dataset["chinese"]), desc="Processing questions"):
        #方法1
        text = format_instruction(src, flag)
        inputs = tokenizer([text], return_tensors="pt", max_length=max_length, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                # do_sample=True, temperature=0.7, top_p=0.9, top_k=40, #50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache = True,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if base_model_name.split('/')[0]=="google":
            translated_text = generated_text.split("model\n")[-1].strip()
        else: #elif base_model_name.split('/')[0]=="meta-llama":
            translated_text = generated_text.split("assistant\n")[-1].strip()
        predictions.append(translated_text)

        #方法2
        text = format_instruction(src, flag, return_flag=True)
        inputs = tokenizer.apply_chat_template(
            [text], add_generation_prompt=True, return_dict=True, # 關鍵：確保回傳的是字典
            return_tensors="pt", max_length=max_length, truncation=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                # do_sample=True, temperature=0.7, top_p=0.9, top_k=40, #50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache = True,
            )
        input_len = inputs.input_ids.shape[1]
        translated_text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
        predictions2.append(translated_text)

    return predictions, predictions2
    

def infer_batch(model, tokenizer, test_dataset):
    predictions = []
    predictions2 = []
    batch_size=5
    for i in tqdm(range(math.ceil(len(test_dataset["chinese"])/batch_size)), total=math.ceil(len(test_dataset["chinese"])/batch_size), desc="Processing questions"):
        #方法1
        texts=[]
        for j in range(batch_size):
            if (i*batch_size+j)>=len(test_dataset["chinese"]): break
            src=test_dataset["chinese"][i*batch_size+j]
            text = format_instruction(src, flag)
            texts.append(text)
        inputs = tokenizer(texts, return_tensors="pt", max_length=max_length, truncation=True, padding=True, padding_side='left').to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                # do_sample=True, temperature=0.7, top_p=0.9, top_k=40, #50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache = True,
            )
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for j in range(len(decoded_outputs)):
            if base_model_name.split('/')[0]=="google": translated_text = decoded_outputs[j].split("model\n")[-1].strip()
            else: translated_text = decoded_outputs[j].split("assistant\n")[-1].strip()
            predictions.append(translated_text)

        #方法2
        texts=[]
        for j in range(batch_size):
            if (i*batch_size+j)>=len(test_dataset["chinese"]): break
            src=test_dataset["chinese"][i*batch_size+j]
            text = format_instruction(src, flag, return_flag=True)
            texts.append(text)
        inputs = tokenizer.apply_chat_template(
            texts, add_generation_prompt=True, return_dict=True, # 關鍵：確保回傳的是字典
            return_tensors="pt", max_length=max_length, truncation=True,
            padding=True, # 關鍵：開啟批量填充
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, #**tokenizer(texts, truncation=True, return_tensors="pt", max_length=max_length).to("cuda"), #
                max_new_tokens=max_length,
                # do_sample=True, temperature=0.7, top_p=0.9, top_k=40, #50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache = True,
            )
        input_len = inputs.input_ids.shape[1]
        decoded_outputs = tokenizer.batch_decode(
            outputs[:, input_len:], # 這裡使用切片，跳過原本的 prompt 部分
            skip_special_tokens=True
        )
        for translated_text in decoded_outputs: predictions2.append(translated_text)
    
    return predictions, predictions2

import time
start = time.process_time()
predictions, predictions2 = infer_batch(model, tokenizer, test_dataset)
# for src, trg1, trg2 in zip(test_dataset["chinese"], predictions, predictions2): print(f"{src}, {trg1}, {trg2}")
end = time.process_time()
print(f"批量翻譯執行時間: {end - start} 秒, 總資料筆數: {len(predictions)}\n")

start = time.process_time()
predictions, predictions2 = infer(model, tokenizer, test_dataset)
# for src, trg1, trg2 in zip(test_dataset["chinese"], predictions, predictions2): print(f"{src}, {trg1}, {trg2}")
end = time.process_time()
print(f"單句翻譯執行時間: {end - start} 秒, 總資料筆數: {len(predictions)}")



# 準備CSV輸出
def save_predictions_to_csv(predictions, predictions2, filename="test_predictions_llama_250730.csv"):
    """將預測結果儲存為CSV檔案"""
    # 創建DataFrame
    data = {
        "id": range(len(predictions)),
        "translation": predictions,
        "translation2": predictions2
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"預測結果已儲存至: {filename}")
    return df
# 儲存預測結果
temp_lst=model_name.split('/')
now = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
filename=f"{temp_lst[-1]}_{mode}_{now}.csv"
df_predictions = save_predictions_to_csv(predictions, predictions2, filename=filename)