#!/usr/bin/env python3
"""
Check if train manifest overlaps with TAT-Vol1 eval or test sets.

Two checks are performed:
  1. Utterance ID exact match (same recording segment)
  2. (speaker, text) match (same speaker, same content, possibly different segment ID)
"""

import json
from pathlib import Path

BASE = Path("/srv/RL_project/TAT-Vol1")


def load_kaldi_map(path: Path) -> dict[str, str]:
    """Read a Kaldi two-column file into {col0: col1}."""
    result = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


def load_eval_data(manifest_dir: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (utt_id_set, (speaker, text) set) for an eval/test manifest dir."""
    utt2spk = load_kaldi_map(manifest_dir / "utt2spk")
    utt2text = load_kaldi_map(manifest_dir / "text")

    utt_ids = set(utt2text.keys())
    spk_text_pairs = {(utt2spk[u], utt2text[u]) for u in utt_ids if u in utt2spk}
    return utt_ids, spk_text_pairs


def load_train_data(manifest_path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (utt_id_set, (speaker, text) set) from train_manifest.jsonl."""
    utt_ids = set()
    spk_text_pairs = set()
    for line in manifest_path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            utt_ids.add(obj["id"])
            spk_text_pairs.add((obj["speaker"], obj["text"]))
    return utt_ids, spk_text_pairs


def report_overlap(label: str, id_overlap: set, content_overlap: set):
    print(f"\n--- {label} ---")
    print(f"  Utterance ID overlap    : {len(id_overlap)}")
    print(f"  (speaker, text) overlap : {len(content_overlap)}")
    if content_overlap:
        print("  Samples (up to 5):")
        for spk, text in sorted(content_overlap)[:5]:
            print(f"    spk={spk}  text={text[:60]}")


def main():
    manifest_path = Path("/srv/RL_project/data/train_manifest.jsonl")
    eval_dir = BASE / "TAT-Vol1-eval_manifest_enhanced"
    test_dir = BASE / "TAT-Vol1-test_manifest_enhanced"

    print("Loading data...")
    train_ids, train_pairs = load_train_data(manifest_path)
    eval_ids, eval_pairs   = load_eval_data(eval_dir)
    test_ids, test_pairs   = load_eval_data(test_dir)

    print(f"  Train : {len(train_ids):>7,} utterances, {len(train_pairs):>7,} (spk, text) pairs")
    print(f"  Eval  : {len(eval_ids):>7,} utterances, {len(eval_pairs):>7,} (spk, text) pairs")
    print(f"  Test  : {len(test_ids):>7,} utterances, {len(test_pairs):>7,} (spk, text) pairs")

    report_overlap("Train ∩ Eval", train_ids & eval_ids, train_pairs & eval_pairs)
    report_overlap("Train ∩ Test", train_ids & test_ids, train_pairs & test_pairs)
    report_overlap("Eval  ∩ Test", eval_ids  & test_ids, eval_pairs  & test_pairs)

    if not any([train_ids & eval_ids, train_ids & test_ids,
                train_pairs & eval_pairs, train_pairs & test_pairs]):
        print("\nNo overlaps found. Train set is clean.")


if __name__ == "__main__":
    main()
