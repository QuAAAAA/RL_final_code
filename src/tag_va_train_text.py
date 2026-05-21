#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
EMPTY_VALUES = {"", "NULL", "null", "None", "none"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    tag: str
    value: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 VA train JSONL 的 Text 轉成含 <aspect>/<opinion> 標籤的文字。"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/va_train/zho_restaurant_train_alltasks.jsonl"),
        help="輸入 JSONL 檔案",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/va_train/zho_restaurant_train_alltasks_tagged.jsonl"),
        help="輸出檔案",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "text"),
        default="jsonl",
        help="jsonl 會保留原欄位並新增 TaggedText；text 只輸出標註後文字",
    )
    parser.add_argument(
        "--include-id",
        action="store_true",
        help="搭配 --format text 時，在每行前面輸出 ID 與 tab",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def overlaps(span: Span, selected: list[Span]) -> bool:
    return any(span.start < item.end and item.start < span.end for item in selected)


def find_next_non_overlapping(
    text: str,
    value: str,
    tag: str,
    selected: list[Span],
) -> Span | None:
    start = 0
    while True:
        index = text.find(value, start)
        if index < 0:
            return None

        span = Span(start=index, end=index + len(value), tag=tag, value=value)
        if not overlaps(span, selected):
            return span

        start = index + 1


def collect_spans(record: dict[str, Any]) -> list[Span]:
    text = str(record.get("Text", ""))
    quadruplets = record.get("Quadruplet", [])
    if not isinstance(quadruplets, list):
        return []

    selected: list[Span] = []
    seen_aspects: set[str] = set()

    for quadruplet in quadruplets:
        if not isinstance(quadruplet, dict):
            continue

        aspect = str(quadruplet.get("Aspect", "")).strip()
        if aspect in EMPTY_VALUES:
            aspect = ""
        if aspect and aspect not in seen_aspects:
            span = find_next_non_overlapping(text, aspect, "aspect", selected)
            if span is None:
                LOGGER.warning("%s 找不到 aspect: %s", record.get("ID", ""), aspect)
            else:
                selected.append(span)
                seen_aspects.add(aspect)

        opinion = str(quadruplet.get("Opinion", "")).strip()
        if opinion in EMPTY_VALUES:
            opinion = ""
        if opinion:
            span = find_next_non_overlapping(text, opinion, "opinion", selected)
            if span is None:
                LOGGER.warning("%s 找不到 opinion: %s", record.get("ID", ""), opinion)
            else:
                selected.append(span)

    return sorted(selected, key=lambda span: (span.start, span.end))


def tag_text(text: str, spans: list[Span]) -> str:
    pieces: list[str] = []
    cursor = 0

    for span in spans:
        pieces.append(text[cursor : span.start])
        pieces.append(f"<{span.tag}>{text[span.start:span.end]}</{span.tag}>")
        cursor = span.end

    pieces.append(text[cursor:])
    return "".join(pieces)


def convert_record(record: dict[str, Any]) -> str:
    text = str(record.get("Text", ""))
    return tag_text(text, collect_spans(record))


def main() -> None:
    args = parse_args()
    configure_logging()

    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number} 不是合法 JSON") from exc

            tagged_text = convert_record(record)
            if args.format == "jsonl":
                record["TaggedText"] = tagged_text
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            elif args.include_id:
                dst.write(f"{record.get('ID', '')}\t{tagged_text}\n")
            else:
                dst.write(tagged_text + "\n")
            count += 1

    LOGGER.info("Wrote %d records to %s", count, output_path)


if __name__ == "__main__":
    main()
