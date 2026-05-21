# tatmoe 評測流程

對 `data/tatmoe_test/condenser/` 資料集跑 Whisper 推理，並以 **Mean Levenshtein Distance（字元級，越小越好）** 評分。

---

## 資料格式

每個說話者資料夾下，每段語音有兩個配對檔案：

```
condenser/KH_KHF1001/
  condenser-KH_KHF1001_A001-1.2.wav          ← 音訊
  condenser-KH_KHF1001_A001-1.2.normalized.txt  ← 標注（含音調數字）
```

標注範例（含音調）：
```
lai5-pin1 tshit4 pah4 peh4 tsap8 ho7 tshiann2-lai5 si3 ho7 kui7-tai5 pan7-li2.
```

音調去除後（模型輸出格式）：
```
lai pin tshit pah peh tsap ho tshiann lai si ho kui tai pan li
```

---

## 步驟

所有指令皆從**專案根目錄**執行。

### Step 1：切分資料 — 建立 manifest

掃描 condenser 目錄，配對 wav / txt，去除音調數字，寫出 `tatmoe/manifest.csv`。

```bash
uv run python tatmoe/prepare_manifest.py
```

可選參數：
```
--data_dir   ./data/tatmoe_test/condenser/   # 資料來源
--output     ./tatmoe/manifest.csv           # 輸出路徑
```

完成後 `manifest.csv` 欄位：`wav_path`, `ground_truth`

---

### Step 2：推理 + 評分

載入 Whisper checkpoint，對 manifest 中所有音檔推理，計算每句 Levenshtein Distance 並輸出 MLD。

```bash
uv run python tatmoe/eval.py
```

可選參數：
```
--model      ./whisper-large-v2-cantonese-finetuned-RawBoost/checkpoint-1800
--manifest   ./tatmoe/manifest.csv
--output     ./tatmoe/results.csv
--batch_size 16
--language   zh
```

完成後 `results.csv` 欄位：`wav_path`, `ground_truth`, `prediction`, `lev_distance`

---

## 工具

| 檔案 | 用途 |
|------|------|
| `strip_tones.py` | 去除 TL 音調數字（可單獨執行或 import） |
| `prepare_manifest.py` | 掃描資料、建立 manifest |
| `eval.py` | 推理 + MLD 評分 |

音調去除測試：
```bash
uv run python tatmoe/strip_tones.py "ma7-si7 thau3-kue3 kong1-tau5 lai5"
# → ma si thau kue kong tau lai
```
