#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

"${PYTHON_BIN}" - <<'PY'
from llm.client import LLMClient

client = LLMClient()
answer = client.chat([
    {"role": "system", "content": "你是一个简洁的连通性测试助手。"},
    {"role": "user", "content": "请只回复：DeepSeek API 连接成功。"},
], max_tokens=64)
print(answer)
PY
