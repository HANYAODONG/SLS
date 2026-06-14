import argparse
import json
from pathlib import Path

from llm.client import LLMClient
from llm.prompts import REPORT_PROMPT, SYSTEM_PROMPT


def generate_report(stats_path, output_path):
    stats = Path(stats_path).read_text(encoding="utf-8")
    client = LLMClient()
    content = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REPORT_PROMPT + "\n\n实验统计 JSON：\n```json\n" + stats + "\n```"},
        ]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content + "\n", encoding="utf-8")
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate an LLM-assisted experiment report.")
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(generate_report(args.stats, args.output))


if __name__ == "__main__":
    main()
