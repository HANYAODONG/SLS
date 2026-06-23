import argparse
import json
import os
import sys
from pathlib import Path

from extensions.asr import audit_text_with_llm, transcribe_with_whisper
from extensions.certification import create_timestamp_record
from extensions.media import extract_audio, extract_keyframes, probe_media
from extensions.report import generate_integrated_report, load_json
from extensions.speaker import enroll_speaker, match_speaker
from extensions.speaker_bridge import run_speaker_command
from extensions.vlm import VisionAuditClient
from analysis.joint_decision_matrix import assess_joint_risk


def print_json(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_preprocess_video(args):
    audio_path = extract_audio(args.video, args.audio_output, sample_rate=args.sample_rate)
    frames = extract_keyframes(args.video, args.frames_dir, fps=args.frame_fps)
    payload = {
        "video": args.video,
        "audio_output": str(audio_path),
        "frames_dir": args.frames_dir,
        "num_frames": len(frames),
        "media_info": probe_media(args.video) if args.probe else None,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_json(payload)


def cmd_certify(args):
    print_json(create_timestamp_record(args.file, args.output))


def cmd_enroll_speaker(args):
    speaker_python = os.environ.get("SPEAKER_PYTHON")
    if speaker_python:
        print_json(
            run_speaker_command(
                speaker_python,
                [
                    "enroll-speaker",
                    "--name",
                    args.name,
                    "--audio",
                    args.audio,
                    "--enrollment",
                    args.enrollment,
                    "--model-name",
                    args.model_name,
                ],
            )
        )
        return
    print_json(enroll_speaker(args.name, args.audio, args.enrollment, model_name=args.model_name))


def cmd_match_speaker(args):
    speaker_python = os.environ.get("SPEAKER_PYTHON")
    if speaker_python:
        print_json(
            run_speaker_command(
                speaker_python,
                [
                    "match-speaker",
                    "--audio",
                    args.audio,
                    "--enrollment",
                    args.enrollment,
                    "--model-name",
                    args.model_name,
                    "--top-k",
                    str(args.top_k),
                ],
            )
        )
        return
    best, all_scores = match_speaker(args.audio, args.enrollment, model_name=args.model_name)
    print_json({"best": best, "all": all_scores[: args.top_k]})


def cmd_asr(args):
    result = transcribe_with_whisper(args.audio, model_name=args.model_name)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_json(result)


def cmd_audit_text(args):
    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    print_json(audit_text_with_llm(text, output_path=args.output))


def cmd_audit_image(args):
    result = VisionAuditClient().audit_image(args.image, prompt=args.prompt)
    payload = {"image": args.image, "audit": result}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_json(payload)


def cmd_report(args):
    payload = load_json(args.input)
    print_json(generate_integrated_report(payload, output_path=args.output))


def cmd_joint_risk(args):
    result = assess_joint_risk(
        voice_similarity=args.voice_similarity,
        fake_probability=args.fake_probability,
        voice_threshold=args.voice_threshold,
        fake_threshold=args.fake_threshold,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_json(result)


def build_parser():
    parser = argparse.ArgumentParser(description="Optional product-style audit extensions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess-video")
    preprocess.add_argument("--video", required=True)
    preprocess.add_argument("--audio-output", required=True)
    preprocess.add_argument("--frames-dir", required=True)
    preprocess.add_argument("--sample-rate", type=int, default=16000)
    preprocess.add_argument("--frame-fps", type=float, default=1.0)
    preprocess.add_argument("--probe", action="store_true")
    preprocess.add_argument("--output")
    preprocess.set_defaults(func=cmd_preprocess_video)

    certify = subparsers.add_parser("certify")
    certify.add_argument("--file", required=True)
    certify.add_argument("--output")
    certify.set_defaults(func=cmd_certify)

    enroll = subparsers.add_parser("enroll-speaker")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--audio", required=True)
    enroll.add_argument("--enrollment", default="artifacts/speaker/enrollment.jsonl")
    enroll.add_argument("--model-name", default="chinese")
    enroll.set_defaults(func=cmd_enroll_speaker)

    match = subparsers.add_parser("match-speaker")
    match.add_argument("--audio", required=True)
    match.add_argument("--enrollment", default="artifacts/speaker/enrollment.jsonl")
    match.add_argument("--model-name", default="chinese")
    match.add_argument("--top-k", type=int, default=5)
    match.set_defaults(func=cmd_match_speaker)

    asr = subparsers.add_parser("asr")
    asr.add_argument("--audio", required=True)
    asr.add_argument("--model-name", default="base")
    asr.add_argument("--output")
    asr.set_defaults(func=cmd_asr)

    audit_text = subparsers.add_parser("audit-text")
    audit_text.add_argument("--text", default="")
    audit_text.add_argument("--text-file")
    audit_text.add_argument("--output")
    audit_text.set_defaults(func=cmd_audit_text)

    audit_image = subparsers.add_parser("audit-image")
    audit_image.add_argument("--image", required=True)
    audit_image.add_argument("--prompt")
    audit_image.add_argument("--output")
    audit_image.set_defaults(func=cmd_audit_image)

    report = subparsers.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument("--output")
    report.set_defaults(func=cmd_report)

    joint = subparsers.add_parser("joint-risk")
    joint.add_argument("--voice-similarity", type=float)
    joint.add_argument("--fake-probability", type=float, required=True)
    joint.add_argument("--voice-threshold", type=float, default=0.70)
    joint.add_argument("--fake-threshold", type=float, default=0.50)
    joint.add_argument("--output")
    joint.set_defaults(func=cmd_joint_risk)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
