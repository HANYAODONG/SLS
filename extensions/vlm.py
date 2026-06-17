import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from llm.client import load_dotenv


def encode_image(image_path):
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("ascii")


class VisionAuditClient:
    def __init__(self, api_key=None, base_url=None, model=None, timeout=60):
        load_dotenv()
        self.api_key = api_key or os.environ.get("VLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.environ.get("VLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("VLM_MODEL")
        self.timeout = timeout
        if not self.api_key or not self.base_url or not self.model:
            raise RuntimeError(
                "VLM is not configured. Set VLM_API_KEY, VLM_BASE_URL and VLM_MODEL "
                "for an OpenAI-compatible vision endpoint."
            )

    def audit_image(self, image_path, prompt=None):
        prompt = prompt or (
            "识别画面中是否存在公众人物、新闻灾害场景、政治敏感场景或其他高风险传播语境。"
            "请用中文输出：敏感/非敏感、主要理由、建议。"
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,{}".format(encode_image(image_path))
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer {}".format(self.api_key),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("VLM API HTTP {}: {}".format(exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("VLM API request failed: {}".format(exc.reason)) from exc
        return data["choices"][0]["message"]["content"]

