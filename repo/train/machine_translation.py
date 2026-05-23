from __future__ import annotations

import argparse
import sys

import requests


DEFAULT_API_URL = "http://tts001.bronci.com.tw:8802/run/predict"
DEFAULT_MODE = "taigi_zh_tw_py"


def translate_text_with_requests(
    input_text: str,
    mode: str = DEFAULT_MODE,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 30.0,
) -> str:
    """
    使用 requests 呼叫翻譯 API 將文字轉換為台語拼音

    Args:
        input_text: 要翻譯的中文文字
        mode: 翻譯模式，預設為 "taigi_zh_tw_py"
        api_url: API URL
        timeout: 請求逾時秒數

    Returns:
        翻譯結果字串
    """
    data = {
        "data": [
            "",  # API對應翻譯模式(純解釋，此欄無作用)
            input_text,  # 輸入文本
            mode,  # 選擇翻譯模式(API)
        ],
        "event_data": None,
        "fn_index": 0,
        "session_hash": "translate_session",
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, json=data, headers=headers, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and result.get("data"):
            return str(result["data"][0])  # 返回翻譯結果
        print("翻譯失敗：API 回應格式不正確", file=sys.stderr)
        return input_text
    except requests.RequestException as e:
        print(f"翻譯請求發生錯誤: {e}", file=sys.stderr)
        return input_text
    except ValueError as e:
        print(f"翻譯失敗：API 回應不是合法 JSON: {e}", file=sys.stderr)
        return input_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Chinese text to Taigi pinyin through the Gradio API.")
    parser.add_argument("text", nargs="*", help="要翻譯的文字；不提供時會從 stdin 讀取")
    parser.add_argument("--mode", default=DEFAULT_MODE, help=f"翻譯模式，預設：{DEFAULT_MODE}")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API URL，預設：{DEFAULT_API_URL}")
    parser.add_argument("--timeout", type=float, default=30.0, help="請求逾時秒數，預設：30")
    args = parser.parse_args()

    input_text = " ".join(args.text).strip()
    if not input_text:
        input_text = sys.stdin.read().strip()
    if not input_text:
        parser.error("請提供要翻譯的文字，或透過 stdin 傳入")

    print(
        translate_text_with_requests(
            input_text,
            mode=args.mode,
            api_url=args.api_url,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
