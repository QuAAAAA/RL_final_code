"""
Evaluate Whisper ASR on TTS-generated wav files from manifest_all.json.
Run from project root: uv run python tatmoe/eval_indextts.

uv run python src/eval_indextts.py \
  --wav_dir /srv/RL_project/repo/outputs/step_1169.pth/ \
  --json /srv/RL_project/repo/outputs/manifest_step1169.json \
  --model_name indextts_step1169 \
  --manifest_out ./src/eval_indextts_manifest_step1169.csv \
  --model ./ASR/whisper-large-v3-turbo-finetuned-pinyin \
  --processor ./ASR/whisper-large-v3-turbo-finetuned-pinyin \
  --output ./src/results_indextts_turbo_pinyin_top_db_20.csv \
  --trim_top_db 20.0

"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from Levenshtein import distance as levenshtein_distance
from jiwer import wer as compute_wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from eval import transcribe_batch
from strip_tones import strip_tl_tones


# (emotion key in manifest, LaTeX column abbreviation)
EMOTION_ORDER = [
    ('neutral', 'Neu.'),
    ('angry', 'Ang.'),
    ('happy', 'Hap.'),
    ('sad', 'Sad'),
    ('surprised', 'Sur.'),
    ('fearful', 'Fea.'),
    ('disgusted', 'Dis.'),
]


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASR on TTS-generated wavs')
    parser.add_argument('--json', default='./data/wav_gen/manifest_indextts.json')
    parser.add_argument('--wav_dir', default='./data/wav_gen/wav/')
    parser.add_argument('--model_name', default='indextts')
    parser.add_argument('--model', default='./whisper-large-v2-cantonese-finetuned-RawBoost/checkpoint-1800')
    parser.add_argument('--processor', default='simonl0909/whisper-large-v2-cantonese')
    parser.add_argument('--manifest_out', default='./data/wav_gen/indextts_manifest.csv')
    parser.add_argument('--output', default='./tatmoe/results_indextts.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--language', default='zh')
    parser.add_argument('--trim_top_db', type=float, default=30.0,
                        help='Silence threshold in dB below peak; LOWER = trim more aggressively. Set <=0 to disable.')
    parser.add_argument('--remove_internal_silence', action='store_true', default=True,
                        help='Use librosa.effects.split to remove silence INSIDE the audio too (not just edges). Default: ON.')
    parser.add_argument('--no_remove_internal_silence', dest='remove_internal_silence', action='store_false')
    args = parser.parse_args()

    with open(args.json, encoding='utf-8') as f:
        entries = json.load(f)

    entries = [e for e in entries if e['model'] == args.model_name]
    print(f'Entries ({args.model_name}): {len(entries)}')

    # The reference for WER must be the text that was ACTUALLY synthesized, i.e. the
    # Taigi pinyin string fed to engine.infer (`synthesis_text` in the manifest, see
    # baseline/indexTTS_gen.py). The `text` field may be the original, un-translated
    # Han characters and would give a meaningless WER. Older manifests have no
    # `synthesis_text` and their `text` is already pinyin, so we fall back to `text`.
    rows = []
    n_from_synth = 0
    for e in entries:
        ref = e.get('synthesis_text') or e['text']
        if e.get('synthesis_text'):
            n_from_synth += 1
        rows.append({
            'index': e['index'],
            'wav_path': os.path.join(args.wav_dir, e['file']),
            'ground_truth_raw': ref,
            'ground_truth': strip_tl_tones(ref),
            'emotion': e.get('emotion', ''),
        })
    print(f'Reference source: synthesis_text={n_from_synth}, text(fallback)={len(rows) - n_from_synth}')

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
    df.to_csv(args.manifest_out, index=False)
    print(f'Manifest saved: {args.manifest_out}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device  : {device}')
    print(f'Loading processor from {args.processor}...')
    processor = AutoProcessor.from_pretrained(args.processor)
    print(f'Loading model from {args.model}...')
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        dtype=torch.float16 if device == 'cuda' else torch.float32,
    ).to(device)
    model.eval()

    trim_top_db = args.trim_top_db if args.trim_top_db and args.trim_top_db > 0 else None
    print(f'Trim top_db          : {trim_top_db}')
    print(f'Remove internal silence: {args.remove_internal_silence}')
    predictions = transcribe_batch(
        df['wav_path'].tolist(), model, processor, device, args.batch_size, args.language,
        trim_top_db=trim_top_db,
        remove_internal_silence=args.remove_internal_silence,
    )

    # Normalize predictions the SAME way as ground_truth (strip_tl_tones already
    # applied to ground_truth above). Without this, a cleaned reference would be
    # scored against a raw hypothesis (tones/hyphens/punct/case) -> inflated WER.
    df['prediction_raw'] = predictions
    df['prediction'] = [strip_tl_tones(p) for p in predictions]
    df['lev_distance'] = [
        levenshtein_distance(p, g)
        for p, g in zip(df['prediction'], df['ground_truth'])
    ]

    refs = df['ground_truth'].tolist()
    hyps = df['prediction'].tolist()

    # Per-utterance WER, used both for the overall average and the per-emotion split.
    df['wer'] = [compute_wer(g, p) for g, p in zip(refs, hyps)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    mld = df['lev_distance'].mean()
    corpus_wer = compute_wer(refs, hyps)
    avg_wer = df['wer'].mean()

    # Per-emotion average WER (NaN -> None when an emotion has no samples).
    per_emotion = {}
    for emo, _ in EMOTION_ORDER:
        sub = df.loc[df['emotion'] == emo, 'wer']
        per_emotion[emo] = sub.mean() if len(sub) else None

    print(f'\n{"=" * 50}')
    print(f'Model        : {args.model_name}')
    print(f'Samples      : {len(df)}')
    print(f'MLD          : {mld:.4f}   (char-level)')
    print(f'Corpus WER   : {corpus_wer:.4f}   ({corpus_wer*100:.2f}%)')
    print(f'Average WER  : {avg_wer:.4f}   ({avg_wer*100:.2f}%)')
    print(f'Output       : {args.output}')
    print(f'{"=" * 50}')

    # Per-emotion sample counts.
    print('Samples per emotion:')
    for emo, _ in EMOTION_ORDER:
        print(f'  {emo:<10}: {(df["emotion"] == emo).sum()}')

    # LaTeX table: Model & Mean & Neu. & Ang. & Hap. & Sad & Sur. & Fea. & Dis.
    def fmt(v):
        return f'{v:.4f}' if v is not None else ''

    header = ' & '.join(
        [r'\textbf{Model}', r'\textbf{Mean}']
        + [rf'\textbf{{{abbr}}}' for _, abbr in EMOTION_ORDER]
    ) + r' \\'
    values = [fmt(avg_wer)] + [fmt(per_emotion[emo]) for emo, _ in EMOTION_ORDER]
    row = f'{args.model_name} & ' + ' & '.join(values) + r' \\'

    print(f'\n{"=" * 50}')
    print('Per-emotion WER (LaTeX):')
    print(header)
    print(r'\midrule')
    print(row)
    print(f'{"=" * 50}')
    print(df[['prediction', 'ground_truth', 'lev_distance']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
