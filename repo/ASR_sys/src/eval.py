"""
Run Whisper inference on tatmoe manifest and compute Mean Levenshtein Distance.
Run from project root: uv run python tatmoe/eval.py
"""
import argparse
import os

import librosa
import numpy as np
import pandas as pd
import torch
from Levenshtein import distance as levenshtein_distance
from jiwer import wer as compute_wer
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def _get_forced_decoder_ids(processor, language):
    tok = processor.tokenizer
    ids = [
        tok.convert_tokens_to_ids(f'<|{language}|>'),
        tok.convert_tokens_to_ids('<|transcribe|>'),
        tok.convert_tokens_to_ids('<|notimestamps|>'),
    ]
    return [(i + 1, token_id) for i, token_id in enumerate(ids)]


def transcribe_batch(audio_paths, model, processor, device, batch_size, language,
                     trim_top_db=None, remove_internal_silence=False):
    model.generation_config.forced_decoder_ids = _get_forced_decoder_ids(processor, language)
    results = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc='Inferring'):
        batch_paths = audio_paths[i: i + batch_size]
        audios = []
        for path in batch_paths:
            try:
                audio, _ = librosa.load(path, sr=16000)
                if trim_top_db is not None:
                    if remove_internal_silence:
                        intervals = librosa.effects.split(audio, top_db=trim_top_db)
                        if len(intervals):
                            audio = np.concatenate([audio[s:e] for s, e in intervals])
                    else:
                        audio, _ = librosa.effects.trim(audio, top_db=trim_top_db)
                    if audio.size == 0:
                        audio = np.zeros(16000, dtype=np.float32)
            except Exception as e:
                print(f'Failed to load {path}: {e}')
                audio = np.zeros(16000, dtype=np.float32)
            audios.append(audio)

        inputs = processor(
            audios,
            sampling_rate=16000,
            return_tensors='pt',
            padding=True,
            return_attention_mask=False,
        )
        input_features = inputs.input_features.to(device=device, dtype=model.dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                max_length=225,
                num_beams=1,
                do_sample=False,
            )

        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        results.extend(transcriptions)

        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Taiwanese ASR on tatmoe')
    parser.add_argument('--model', default='./whisper-large-v2-cantonese-finetuned-RawBoost/checkpoint-1800')
    parser.add_argument('--processor', default='simonl0909/whisper-large-v2-cantonese',
                        help='Processor/tokenizer source (base model HF ID or local path)')
    parser.add_argument('--manifest', default='./tatmoe/manifest.csv')
    parser.add_argument('--output', default='./tatmoe/results.csv')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--language', default='zh')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device  : {device}')

    df = pd.read_csv(args.manifest)
    print(f'Samples : {len(df)}')

    print(f'Loading model from {args.model}...')
    print(f'Loading processor from {args.processor}...')
    processor = AutoProcessor.from_pretrained(args.processor)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        dtype=torch.float16 if device == 'cuda' else torch.float32,
    ).to(device)
    model.eval()

    predictions = transcribe_batch(
        df['wav_path'].tolist(), model, processor, device, args.batch_size, args.language
    )

    df['prediction'] = predictions
    df['lev_distance'] = [
        levenshtein_distance(pred, gt)
        for pred, gt in zip(predictions, df['ground_truth'])
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    mld = df['lev_distance'].mean()
    wer = compute_wer(df['ground_truth'].tolist(), df['prediction'].tolist())

    print(f'\n{"=" * 50}')
    print(f'Samples : {len(df)}')
    print(f'MLD     : {mld:.4f}   (char-level, Kaggle metric)')
    print(f'WER     : {wer:.4f}   (syllable-level)')
    print(f'Min LD  : {df["lev_distance"].min()}')
    print(f'Max LD  : {df["lev_distance"].max()}')
    print(f'Output  : {args.output}')
    print(f'{"=" * 50}')
    print(df[['prediction', 'ground_truth', 'lev_distance']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
