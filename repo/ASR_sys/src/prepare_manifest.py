"""
Scan data/tatmoe_test/condenser/ and write tatmoe/manifest.csv.

Each row: wav_path, ground_truth (tone-stripped TL text)
Run from project root: uv run python tatmoe/prepare_manifest.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from strip_tones import strip_tl_tones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/tatmoe_test/condenser/')
    parser.add_argument('--output', default='./tatmoe/manifest.csv')
    args = parser.parse_args()

    rows = []
    missing_wav = 0
    for root, _, files in os.walk(args.data_dir):
        for fname in sorted(files):
            if not fname.endswith('.normalized.txt'):
                continue
            txt_path = os.path.join(root, fname)
            wav_path = txt_path.replace('.normalized.txt', '.wav')
            if not os.path.exists(wav_path):
                missing_wav += 1
                continue
            with open(txt_path, encoding='utf-8') as f:
                gt = strip_tl_tones(f.read().strip())
            rows.append({'wav_path': wav_path, 'ground_truth': gt})

    rows.sort(key=lambda r: r['wav_path'])
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f'Pairs found  : {len(df)}')
    print(f'Missing wav  : {missing_wav}')
    print(f'Manifest     : {args.output}')
    print(df.head(5).to_string(index=False))


if __name__ == '__main__':
    main()
