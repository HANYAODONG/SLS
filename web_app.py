import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from analysis.score_stats import compute_df_stats
from llm.client import LLMClient
from llm.prompts import SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"

EXPERIMENTS = {
    "df_20000": {
        "name": "ASVspoof 2021 DF first20000",
        "score": "scores/scores_DF_20000.txt",
        "metadata": "keys/DF_20000/CM/trial_metadata.txt",
        "protocol": "database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.first20000.trl.txt",
        "phase": "eval",
        "command": "bash scripts/eval_df_20000.sh && bash scripts/eer_df_20000.sh",
        "reported_full_df_eer": 1.92,
    },
    "df_5000": {
        "name": "ASVspoof 2021 DF first5000",
        "score": "scores/scores_DF_5000.txt",
        "metadata": "keys/DF_5000/CM/trial_metadata.txt",
        "protocol": "database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.first5000.trl.txt",
        "phase": "eval",
        "command": "bash scripts/eval_df_5000.sh && bash scripts/eer_df_5000.sh",
        "reported_full_df_eer": 1.92,
    },
}


def json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler, text, status=200, content_type="text/plain; charset=utf-8"):
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def get_experiment_summary(experiment_id):
    config = EXPERIMENTS[experiment_id]
    stats = compute_df_stats(
        ROOT / config["score"],
        ROOT / config["metadata"],
        phase=config["phase"],
        experiment=experiment_id,
    )
    stats.update(
        {
            "name": config["name"],
            "protocol": config["protocol"],
            "command": config["command"],
            "reported_full_df_eer": config["reported_full_df_eer"],
            "is_subset": True,
        }
    )
    return stats


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/experiments":
            json_response(self, {"experiments": list(EXPERIMENTS.keys())})
            return
        if parsed.path == "/api/summary":
            experiment_id = "df_20000"
            if parsed.query:
                for part in parsed.query.split("&"):
                    if part.startswith("experiment="):
                        experiment_id = part.split("=", 1)[1]
            if experiment_id not in EXPERIMENTS:
                json_response(self, {"error": "Unknown experiment"}, status=404)
                return
            json_response(self, get_experiment_summary(experiment_id))
            return
        self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            json_response(self, {"error": "Not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        question = (data.get("question") or "").strip()
        experiment_id = data.get("experiment") or "df_20000"
        if not question:
            json_response(self, {"error": "Question is required"}, status=400)
            return
        if experiment_id not in EXPERIMENTS:
            json_response(self, {"error": "Unknown experiment"}, status=404)
            return
        try:
            summary = get_experiment_summary(experiment_id)
            client = LLMClient()
            answer = client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "项目当前实验统计如下：\n```json\n"
                            + json.dumps(summary, ensure_ascii=False, indent=2)
                            + "\n```\n\n用户问题："
                            + question
                        ),
                    },
                ]
            )
            json_response(self, {"answer": answer, "summary": summary})
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)

    def serve_static(self, path):
        if path == "/":
            path = "/index.html"
        requested = (WEB_ROOT / path.lstrip("/")).resolve()
        if not str(requested).startswith(str(WEB_ROOT.resolve())) or not requested.is_file():
            text_response(self, "Not found", status=404)
            return
        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        body = requested.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local LLM web assistant.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument(
        "--no-port-fallback",
        action="store_true",
        help="Fail immediately if the selected port is already in use.",
    )
    return parser.parse_args()


def create_server(host, port, allow_fallback=True, attempts=20):
    last_error = None
    for candidate in range(port, port + (attempts if allow_fallback else 1)):
        try:
            return candidate, ThreadingHTTPServer((host, candidate), AppHandler)
        except OSError as exc:
            last_error = exc
            if exc.errno != 98:
                raise
    raise last_error


def main():
    args = parse_args()
    port, server = create_server(args.host, args.port, allow_fallback=not args.no_port_fallback)
    if port != args.port:
        print(f"Port {args.port} is in use; switched to {port}.")
    print(f"LLM web assistant running at http://{args.host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
