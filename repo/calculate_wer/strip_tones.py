import argparse
import json
import re
import sys
from pathlib import Path

# Punctuation to replace with space (curly quotes via unicode escapes to avoid encoding issues)
_PUNCT = re.compile(r"[.,!?;:'\"%、，。？！；：「」（）()“”‘’…]")


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "outputs" / "manifest_indextts.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "manifest_indextts_toneless.json"


def strip_tl_tones(text: str) -> str:
    """Strip tone digits, hyphens, and punctuation from TL romanization text."""
    text = re.sub(r"\d", "", text)
    text = text.replace("-", " ")
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def strip_manifest_text(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} 應該是 JSON list")

    for item in data:
        if isinstance(item, dict) and "text" in item:
            item["text"] = strip_tl_tones(str(item["text"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return len(data)


def parse_args():
    parser = argparse.ArgumentParser(description="去除台羅 manifest/text 裡的音調數字")
    parser.add_argument("text", nargs="*", help="直接提供文字時，只輸出去音調後文字")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="輸入 manifest JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="輸出 toneless manifest JSON")
    parser.add_argument("--stdin", action="store_true", help="逐行讀取 stdin 並輸出去音調後文字")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.text:
        print(strip_tl_tones(" ".join(args.text)))
    elif args.stdin:
        for line in sys.stdin:
            print(strip_tl_tones(line.rstrip("\n")))
    else:
        count = strip_manifest_text(args.input, args.output)
        print(f"已處理 {count} 筆，輸出：{args.output}")


if __name__ == "__main__":
    main()
