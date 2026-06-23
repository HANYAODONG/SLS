import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.audio_utils import load_audio_16k
from explainability.evidence_builder import build_evidence
from explainability.frequency_ablation import frequency_ablation
from explainability.llm_report import template_report
from explainability.model_adapter import build_adapter
from explainability.occlusion import time_occlusion
from explainability.stability import stability_analysis
from explainability.validator import validate_report


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_hybrid_args(parser):
    parser.add_argument("--model-type", default="original", choices=["original", "hybrid"])
    parser.add_argument("--use-stat-sls", type=int, default=1)
    parser.add_argument("--stat-sls-use-std", type=int, default=1)
    parser.add_argument("--use-swiglu", type=int, default=1)
    parser.add_argument("--pooling-type", default="cgta", choices=["maxpool", "temporal", "cgta"])
    parser.add_argument("--cgta-use-std", type=int, default=1)
    parser.add_argument("--cgta-stat-residual", type=int, default=1)
    parser.add_argument("--hybrid-hidden-dim", type=int, default=128)
    parser.add_argument("--hybrid-dropout", type=float, default=0.1)


def main():
    parser = argparse.ArgumentParser(description="Generate zero-training explainability evidence for one audio file.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", default="MMpaper_model.pth")
    parser.add_argument("--xlsr-checkpoint", default="xlsr2_300m.pt")
    parser.add_argument("--config", default="configs/explainability.json")
    parser.add_argument("--output-dir", default="artifacts/reports")
    parser.add_argument("--device", default=None)
    add_hybrid_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = build_adapter(args)
    audio = load_audio_16k(args.audio, sample_rate=int(config.get("sample_rate", 16000)))
    prediction = adapter.predict_audio(audio)
    details = adapter.details_audio(audio)
    base_fake_probability = prediction["fake_probability"]

    time_rows = time_occlusion(adapter, audio, base_fake_probability, config)
    frequency_rows = frequency_ablation(adapter, audio, base_fake_probability, config)
    stability = stability_analysis(adapter, audio, config)
    evidence = build_evidence(
        args.audio,
        audio,
        prediction,
        details,
        time_rows,
        frequency_rows,
        stability,
        config,
    )
    report = template_report(evidence)
    validation = validate_report(report)
    if not validation["ok"]:
        report = template_report(evidence)
        validation = validate_report(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.audio).stem
    evidence_path = output_dir / "{}_evidence.json".format(stem)
    report_path = output_dir / "{}_report.txt".format(stem)
    evidence["report_validation"] = validation
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "evidence_path": str(evidence_path),
                "report_path": str(report_path),
                "fake_probability": base_fake_probability,
                "hidden_states_shape": details.get("hidden_states_shape"),
                "fused_sequence_shape": details.get("fused_sequence_shape"),
                "report_validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
