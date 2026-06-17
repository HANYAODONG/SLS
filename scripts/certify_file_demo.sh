#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
TARGET_FILE="${1:?Usage: bash scripts/certify_file_demo.sh path/to/file}"
CASE_ID="${CASE_ID:-demo_case}"

"${PYTHON_BIN}" extension_audit.py certify \
  --file "${TARGET_FILE}" \
  --output "artifacts/audit/${CASE_ID}/timestamp.json"
