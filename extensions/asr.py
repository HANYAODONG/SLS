import json
from pathlib import Path

from llm.client import LLMClient


def transcribe_with_whisper(audio_path, model_name="base"):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is not installed. Install optional dependency `openai-whisper` "
            "before using ASR transcription."
        ) from exc
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return {
        "audio_path": str(audio_path),
        "model": model_name,
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
    }


def audit_text_with_llm(text, output_path=None):
    prompt = (
        "请作为社交平台内容安全审核助手，判断下面的语音转写文本是否包含诈骗诱导、"
        "虚假指令、敏感公共事件误导、冒充公众人物发言等风险。请输出风险等级、理由和处置建议。\n\n"
        "转写文本：\n{}".format(text)
    )
    client = LLMClient()
    answer = client.chat(
        [
            {"role": "system", "content": "你是音频深度伪造内容审计助手，回答必须中文、克制、可用于项目演示。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    result = {"text": text, "audit": answer}
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result

