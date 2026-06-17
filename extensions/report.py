import json
from pathlib import Path

from llm.client import LLMClient


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_integrated_report(payload, output_path=None):
    prompt = (
        "请基于下面的多路审计结果生成中文风险报告。报告应包含：总体风险等级、"
        "音频伪造检测结论、声纹匹配结论、ASR 内容风险、画面语义风险、证据固化信息、"
        "建议处置动作。不要编造输入中不存在的事实。\n\n"
        "审计数据：\n```json\n{}\n```".format(json.dumps(payload, ensure_ascii=False, indent=2))
    )
    client = LLMClient()
    report = client.chat(
        [
            {"role": "system", "content": "你是深度伪造音视频离线审计报告助手，回答必须中文、结构清晰、谨慎。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1800,
    )
    result = {"input": payload, "report": report}
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result

