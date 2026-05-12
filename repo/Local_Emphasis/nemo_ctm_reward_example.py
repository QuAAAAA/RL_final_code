"""Example: compute local emphasis reward from NeMo forced-aligner CTM output.

Run after:
    bash repo/NeMo/tools/nemo_forced_aligner/run_align.sh

Example:
    repo/NeMo/.venv/bin/python repo/Local_Emphasis/nemo_ctm_reward_example.py \
        --quote-index all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo.Local_Emphasis import LocalEmphasisReward
from repo.Local_Emphasis.local_emphasis_reward import WordBoundary


QUOTE_CHARS = {'"', "“", "”"}


def token_has_quote(token: str) -> bool:
    """Return True when a CTM token contains a quote marker."""

    return any(char in token for char in QUOTE_CHARS)


def strip_quote_chars(token: str) -> str:
    """Remove quote markers while preserving any real token text."""

    for char in QUOTE_CHARS:
        token = token.replace(char, "")
    return token


def read_ctm_words(ctm_path: Path) -> List[WordBoundary]:
    """Read NeMo word-level CTM as (word, start_sec, end_sec) boundaries."""

    boundaries: List[WordBoundary] = []
    for line_no, line in enumerate(ctm_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Malformed CTM line {line_no}: {line}")
        start = float(parts[2])
        duration = float(parts[3])
        word = parts[4]
        boundaries.append((word, start, start + duration))
    if not boundaries:
        raise ValueError(f"No word boundaries found in {ctm_path}")
    return boundaries


def remove_quote_tokens_and_find_targets(
    boundaries: List[WordBoundary],
) -> tuple[List[WordBoundary], List[int]]:
    """Drop quote-only CTM tokens and return target indices inside quotes.

    NeMo CTM can include the emphasis markers themselves as word tokens, e.g.
    ``"`` and ``".``. The reward should be computed over spoken words, while the
    selected target should be the word enclosed by quotes in the original text.
    """

    cleaned: List[WordBoundary] = []
    quote_target_indices: List[int] = []
    inside_quote = False

    for word, start, end in boundaries:
        opens_or_closes_quote = token_has_quote(word)
        spoken_word = strip_quote_chars(word).strip()

        if spoken_word and spoken_word not in {".", ",", "?", "!", ":", ";"}:
            cleaned_idx = len(cleaned)
            cleaned.append((spoken_word, start, end))
            if inside_quote:
                quote_target_indices.append(cleaned_idx)

        if opens_or_closes_quote:
            inside_quote = not inside_quote

    if not cleaned:
        raise ValueError("No spoken word boundaries remain after removing quotes.")
    return cleaned, quote_target_indices


def exclude_other_quoted_targets(
    boundaries: List[WordBoundary],
    quoted_target_indices: List[int],
    target_idx: int,
) -> tuple[List[WordBoundary], int]:
    """Remove other quoted target words and remap the current target index."""

    quoted_set = set(quoted_target_indices)
    filtered: List[WordBoundary] = []
    remapped_target_idx: int | None = None

    for original_idx, boundary in enumerate(boundaries):
        if original_idx in quoted_set and original_idx != target_idx:
            continue
        if original_idx == target_idx:
            remapped_target_idx = len(filtered)
        filtered.append(boundary)

    if remapped_target_idx is None:
        raise ValueError(f"target_idx={target_idx} was removed or does not exist.")
    return filtered, remapped_target_idx


def load_wav(audio_path: Path) -> tuple[torch.Tensor, int]:
    """Load wav with soundfile and convert multi-channel audio to mono."""

    wav_np, sr = sf.read(audio_path, dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    return torch.from_numpy(wav_np), int(sr)


def parse_args() -> argparse.Namespace:
    default_asset = PROJECT_ROOT / "repo/NeMo/tools/nemo_forced_aligner/asset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=default_asset / "test.wav")
    parser.add_argument(
        "--ctm",
        type=Path,
        default=default_asset / "output/ctm/words/test.ctm",
    )
    parser.add_argument(
        "--target-idx",
        type=int,
        default=None,
        help="Target index after quote/punctuation tokens are removed.",
    )
    parser.add_argument(
        "--quote-index",
        default="all",
        help=(
            "Which quoted word to target when --target-idx is not provided. "
            "Use 'all' to sum rewards over every quoted word."
        ),
    )
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument(
        "--preset",
        choices=["paper_baseline", "rl_robust"],
        default="rl_robust",
        help="Reward preset to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav, sr = load_wav(args.audio)
    raw_boundaries = read_ctm_words(args.ctm)
    boundaries, quoted_target_indices = remove_quote_tokens_and_find_targets(
        raw_boundaries
    )
    if args.target_idx is not None:
        target_indices = [args.target_idx]
    else:
        if not quoted_target_indices:
            raise ValueError(
                "No quoted target word found in CTM. Pass --target-idx explicitly."
            )
        if args.quote_index == "all":
            target_indices = quoted_target_indices
        else:
            try:
                quote_index = int(args.quote_index)
            except ValueError as exc:
                raise ValueError("--quote-index must be an integer or 'all'.") from exc
            if quote_index < 0 or quote_index >= len(quoted_target_indices):
                raise ValueError(
                    f"--quote-index={quote_index} is out of range; found "
                    f"{len(quoted_target_indices)} quoted target word(s)."
                )
            target_indices = [quoted_target_indices[quote_index]]

    reward_fn = LocalEmphasisReward(
        preset=args.preset,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    target_rewards = []
    for target_idx in target_indices:
        filtered_boundaries, remapped_target_idx = exclude_other_quoted_targets(
            boundaries,
            quoted_target_indices,
            target_idx,
        )
        reward = reward_fn(wav, sr, filtered_boundaries, remapped_target_idx)
        reward["target_idx"] = remapped_target_idx
        reward["original_target_idx"] = target_idx
        reward["excluded_quoted_target_indices"] = [
            idx for idx in quoted_target_indices if idx != target_idx
        ]
        target_rewards.append(reward)

    total_mean = sum(float(reward["total"]) for reward in target_rewards) / len(
        target_rewards
    )
    output = {
        "total": float(total_mean),
        "aggregation": "mean",
        "num_targets": len(target_rewards),
        "preset": args.preset,
        "target_rewards": target_rewards,
        "quoted_target_indices": quoted_target_indices,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
