import argparse
import cgi
import json
import mimetypes
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from analysis.score_stats import compute_df_stats
from analysis.joint_decision_matrix import assess_joint_risk
from extensions.speaker import match_speaker
from extensions.speaker_bridge import run_speaker_command
from llm.client import LLMClient
from llm.prompts import SYSTEM_PROMPT
from single_audio_infer import infer_audio


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
RECORDING_ROOT = ROOT / "samples" / "recordings"
PROCESSED_ROOT = ROOT / "samples" / "processed"

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
    "wild_20000": {
        "name": "In-the-Wild first20000",
        "score": "scores/scores_Wild_20000.txt",
        "metadata": "keys/Wild_20000/trial_metadata.txt",
        "protocol": "database/ASVspoof_DF_cm_protocols/in_the_wild.first20000.eval.txt",
        "phase": "eval",
        "command": "bash scripts/eval_wild_20000.sh && bash scripts/eer_wild_20000.sh",
    },
    "wild_tiny10": {
        "name": "In-the-Wild tiny10",
        "score": "scores/scores_Wild_tiny10.txt",
        "metadata": "keys/Wild_tiny10/trial_metadata.txt",
        "protocol": "database/ASVspoof_DF_cm_protocols/in_the_wild.tiny10.eval.txt",
        "phase": "eval",
        "command": "bash scripts/eval_wild_tiny10.sh && bash scripts/eer_wild_tiny10.sh",
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
            "is_subset": True,
        }
    )
    if "reported_full_df_eer" in config:
        stats["reported_full_df_eer"] = config["reported_full_df_eer"]
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
        if parsed.path == "/api/analyze-recording":
            self.handle_recording_analysis()
            return
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

    def handle_recording_analysis(self):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            json_response(self, {"error": "multipart/form-data is required"}, status=400)
            return

        try:
            RECORDING_ROOT.mkdir(parents=True, exist_ok=True)
            PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            if "audio" not in form:
                json_response(self, {"error": "audio file is required"}, status=400)
                return
            item = form["audio"]
            voice_similarity = None
            if "voice_similarity" in form:
                value = (form["voice_similarity"].value or "").strip()
                if value:
                    voice_similarity = float(value)
            raw_name = Path(item.filename or "recording.webm").name
            suffix = Path(raw_name).suffix or ".webm"
            sample_id = uuid.uuid4().hex
            saved_path = RECORDING_ROOT / "{}{}".format(sample_id, suffix)
            with open(saved_path, "wb") as handle:
                handle.write(item.file.read())

            converted_path = PROCESSED_ROOT / "{}_16k.wav".format(sample_id)
            result = infer_audio(
                saved_path,
                converted_path=converted_path,
                model_path=os.environ.get("SLS_MODEL_PATH", "MMpaper_model.pth"),
                xlsr_checkpoint=os.environ.get("XLSR_CHECKPOINT", "xlsr2_300m.pt"),
            )
            speaker_result = self.try_match_speaker(converted_path)
            if speaker_result and voice_similarity is None:
                voice_similarity = speaker_result["best"]["similarity"]
            result["speaker_match"] = speaker_result
            result["joint_decision"] = assess_joint_risk(
                voice_similarity=voice_similarity,
                fake_probability=result.get("probabilities", {}).get("fake_probability"),
                voice_threshold=float(os.environ.get("JOINT_VOICE_THRESHOLD", "0.70")),
                fake_threshold=float(os.environ.get("JOINT_FAKE_THRESHOLD", "0.50")),
            )
            result["sample_id"] = sample_id
            result["original_filename"] = raw_name
            json_response(self, result)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)

    def try_match_speaker(self, audio_path):
        enrollment = os.environ.get("SPEAKER_ENROLLMENT")
        if not enrollment:
            return None
        if not Path(enrollment).is_file():
            return {"error": "SPEAKER_ENROLLMENT not found: {}".format(enrollment)}
        model_name = os.environ.get("SPEAKER_MODEL_NAME", "chinese")
        try:
            speaker_python = os.environ.get("SPEAKER_PYTHON")
            if speaker_python:
                return run_speaker_command(
                    speaker_python,
                    [
                        "match-speaker",
                        "--audio",
                        str(audio_path),
                        "--enrollment",
                        enrollment,
                        "--model-name",
                        model_name,
                        "--top-k",
                        "5",
                    ],
                )
            best, all_scores = match_speaker(audio_path, enrollment, model_name=model_name)
            return {"best": best, "all": all_scores[:5]}
        except Exception as exc:
            return {"error": str(exc)}

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
