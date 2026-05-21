import os
import time
import argparse
import logging
import multiprocessing as mp
from pathlib import Path

# ── 設定 ──────────────────────────────────────────────────────────────────────
# 原始音檔根目錄
WAV_ROOT    = "/nfs/DGX2/Workspace/chipsrin/Taigi_Drama_10000hr"
# 辨識結果存放目錄 (建議與來源目錄分開，以免混亂)
RESULT_ROOT = "/nfs/DGX2/Workspace/chipsrin/Taigi_Drama_10000hr_asr_result"
# Log 存放位置
LOG_PATH    = "./asr_infer_taigi.log"

# 模型路徑 (學長提供的路徑)
MODEL_PATH  = "/mnt/usr1/ethan1017/Taiwanese_ASR/faster_whisper/ASR-Taigi-TW-Faster"

DEFAULT_GPUS     = "0,1,2,3"         # 使用 4 張 GPU
BEAM_SIZE        = 5
LANGUAGE         = "zh"              # 台語辨識通常在 Whisper 中對應 zh
COMPUTE_TYPE     = "float16"         # 若顯卡支援，float16 速度最快
QUEUE_MAXSIZE    = 1000              # Queue 緩衝大小

# ── 遞迴掃描待辨識清單 ───────────────────────────────────────────────────────
def iter_pending(wav_root, result_root):
    """使用 os.walk 遞迴掃描所有層級的 .wav 檔，並計算對應的輸出路徑"""
    for root, dirs, files in os.walk(wav_root):
        for fname in files:
            if fname.lower().endswith('.wav'):
                wav_path = os.path.join(root, fname)
                
                # 計算相對路徑，用來在 result_root 重建資料夾結構
                rel_path = os.path.relpath(wav_path, wav_root)
                stem = os.path.splitext(rel_path)[0]
                res_path = os.path.join(result_root, stem + '.txt')
                
                # 斷點續傳：若 .txt 已存在則跳過
                if os.path.exists(res_path):
                    continue
                
                yield wav_path, res_path

# ── Worker：每張 GPU 一個 process ─────────────────────────────────────────────
def worker(gpu_id, model_path, queue, counter, lock, log_queue):
    """從 queue 取 (wav_path, res_path) 並進行辨識"""
    from faster_whisper import WhisperModel

    try:
        # 載入學長指定的模型路徑
        model = WhisperModel(
            model_path,
            device="cuda",
            device_index=gpu_id,
            compute_type=COMPUTE_TYPE,
        )
        log_queue.put(f"[GPU {gpu_id}] 模型載入完成：{model_path}")
    except Exception as e:
        log_queue.put(f"[GPU {gpu_id}] 模型載入失敗：{e}")
        return

    while True:
        item = queue.get()
        if item is None:
            break

        wav_path, res_path = item
        try:
            segments, _ = model.transcribe(
                wav_path,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=True,
            )
            # 將所有片段合併為一段文字
            text = ' '.join(seg.text.strip() for seg in segments).strip()

            # 確保輸出目錄存在
            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            with open(res_path, 'w', encoding='utf-8') as f:
                f.write(text + '\n')

            with lock:
                counter.value += 1

        except Exception as e:
            log_queue.put(f"[GPU {gpu_id}] 錯誤檔名 {os.path.basename(wav_path)}: {e}")

# ── Log writer ──────────────────────────────────────────────────────────────
def log_writer(log_queue, log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(),
        ]
    )
    while True:
        msg = log_queue.get()
        if msg is None:
            break
        logging.info(msg)

# ── 主程式 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="台語劇集 10000hr 大規模 ASR 辨識")
    parser.add_argument('--gpus', default=DEFAULT_GPUS, help='使用哪些 GPU (預設 0,1,2,3)')
    parser.add_argument('--model', default=MODEL_PATH, help='模型路徑')
    args = parser.parse_args()

    gpu_list = [int(g) for g in args.gpus.split(',')]
    
    print("="*50)
    print(f"啟動並行辨識任務")
    print(f"GPU 列表   : {gpu_list}")
    print(f"模型路徑   : {args.model}")
    print(f"音檔根目錄 : {WAV_ROOT}")
    print(f"結果儲存處 : {RESULT_ROOT}")
    print("="*50)

    # 確保使用的啟動模式為 spawn (CUDA 必需)
    mp.set_start_method('spawn', force=True)

    queue     = mp.Queue(maxsize=QUEUE_MAXSIZE)
    counter   = mp.Value('i', 0)
    lock      = mp.Lock()
    log_queue = mp.Queue()

    # 1. 啟動 Log Writer
    log_proc = mp.Process(target=log_writer, args=(log_queue, LOG_PATH), daemon=True)
    log_proc.start()

    # 2. 啟動 GPU Workers
    workers = []
    for gid in gpu_list:
        p = mp.Process(
            target=worker,
            args=(gid, args.model, queue, counter, lock, log_queue),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # 3. 主程式：遞迴掃描並填入任務
    total_enqueued = 0
    t0 = time.time()
    
    try:
        for wav_path, res_path in iter_pending(WAV_ROOT, RESULT_ROOT):
            queue.put((wav_path, res_path))
            total_enqueued += 1
            
            # 每送出 500 筆更新一次終端進度
            if total_enqueued % 500 == 0:
                done = counter.value
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                print(f" >> 已送出: {total_enqueued} | 已完成: {done} | 速度: {speed:.2f} 檔/秒", end='\r')

    except KeyboardInterrupt:
        print("\n使用者手動停止，正在關閉 worker...")

    # 4. 結束處理
    for _ in workers:
        queue.put(None)

    for p in workers:
        p.join()

    log_queue.put(None)
    log_proc.join()

    elapsed = time.time() - t0
    print(f"\n\n任務結束！")
    print(f"總共完成辨識: {counter.value} 個音檔")
    print(f"總耗時: {elapsed/3600:.2f} 小時")

if __name__ == '__main__':
    main()