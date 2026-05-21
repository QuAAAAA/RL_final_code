import re
import sys

# Punctuation to replace with space (curly quotes via unicode escapes to avoid encoding issues)
_PUNCT = re.compile(r"[.,!?;:'\"" + "「」（）()“”‘’]")


def strip_tl_tones(text: str) -> str:
    """Strip tone numbers (1-8) and hyphens from TL romanization text."""
    text = re.sub(r'[1-8]', '', text)
    text = text.replace('-', ' ')
    text = _PUNCT.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(strip_tl_tones(' '.join(sys.argv[1:])))
    else:
        for line in sys.stdin:
            print(strip_tl_tones(line.rstrip('\n')))
