import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.score_stats import read_metadata, read_scores, row_label, row_phase, row_utt_id


def collect_labeled_scores(score_path, metadata_path, phase="eval", positive_label="bonafide"):
    scores = read_scores(score_path)
    metadata_rows = read_metadata(metadata_path)
    has_phase_column = any(len(row) > 7 for row in metadata_rows)
    y_true = []
    y_score = []
    rows = []

    for row in metadata_rows:
        if has_phase_column and row_phase(row) != phase:
            continue
        utt_id = row_utt_id(row)
        if utt_id not in scores:
            continue
        label = row_label(row)
        if label not in ("bonafide", "spoof"):
            continue
        score = scores[utt_id]
        y_true.append(1 if label == positive_label else 0)
        y_score.append(score)
        rows.append({"utt_id": utt_id, "label": label, "score": score})

    if not y_true:
        raise ValueError("No scored rows found for ROC/AUC calculation.")
    if len(set(y_true)) != 2:
        raise ValueError("ROC/AUC needs both positive and negative labels.")
    return np.asarray(y_true), np.asarray(y_score), rows


def roc_curve_points(y_true, y_score):
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    tps = 0
    fps = 0
    points = [{"threshold": None, "fpr": 0.0, "tpr": 0.0}]

    last_score = None
    for label, score in zip(y_true, y_score):
        if last_score is not None and score != last_score:
            points.append(
                {
                    "threshold": float(last_score),
                    "fpr": float(fps / negatives),
                    "tpr": float(tps / positives),
                }
            )
        if label == 1:
            tps += 1
        else:
            fps += 1
        last_score = score

    points.append(
        {
            "threshold": float(last_score),
            "fpr": float(fps / negatives),
            "tpr": float(tps / positives),
        }
    )
    points.append({"threshold": None, "fpr": 1.0, "tpr": 1.0})
    return points


def auc_from_points(points):
    fpr = np.asarray([point["fpr"] for point in points], dtype=float)
    tpr = np.asarray([point["tpr"] for point in points], dtype=float)
    order = np.argsort(fpr, kind="mergesort")
    return float(np.trapz(tpr[order], fpr[order]))


def write_csv(points, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "fpr", "tpr"])
        writer.writeheader()
        for point in points:
            writer.writerow(point)


def write_svg(points, output_path, width=720, height=520, margin=56):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_w = width - margin * 2
    plot_h = height - margin * 2

    def xy(point):
        x = margin + point["fpr"] * plot_w
        y = height - margin - point["tpr"] * plot_h
        return x, y

    polyline = " ".join("{:.2f},{:.2f}".format(*xy(point)) for point in points)
    diagonal = "{},{} {},{}".format(margin, height - margin, width - margin, margin)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#1f2937" stroke-width="2"/>
  <line x1="{margin}" y1="{height - margin}" x2="{margin}" y2="{margin}" stroke="#1f2937" stroke-width="2"/>
  <polyline points="{diagonal}" fill="none" stroke="#9ca3af" stroke-width="2" stroke-dasharray="6 6"/>
  <polyline points="{polyline}" fill="none" stroke="#1769e0" stroke-width="3"/>
  <text x="{width / 2}" y="{height - 14}" text-anchor="middle" font-family="sans-serif" font-size="16">False Positive Rate</text>
  <text x="18" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="16" transform="rotate(-90 18 {height / 2})">True Positive Rate</text>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">ROC Curve</text>
  <text x="{margin}" y="{height - margin + 26}" font-family="sans-serif" font-size="12">0</text>
  <text x="{width - margin - 8}" y="{height - margin + 26}" font-family="sans-serif" font-size="12">1</text>
  <text x="{margin - 24}" y="{height - margin + 4}" font-family="sans-serif" font-size="12">0</text>
  <text x="{margin - 24}" y="{margin + 4}" font-family="sans-serif" font-size="12">1</text>
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")


def compute_roc_auc(score_path, metadata_path, phase="eval", positive_label="bonafide"):
    y_true, y_score, rows = collect_labeled_scores(
        score_path,
        metadata_path,
        phase=phase,
        positive_label=positive_label,
    )
    points = roc_curve_points(y_true, y_score)
    auc = auc_from_points(points)
    return {
        "score_file": str(score_path),
        "metadata_file": str(metadata_path),
        "phase": phase,
        "positive_label": positive_label,
        "num_samples": int(len(y_true)),
        "num_positive": int(np.sum(y_true == 1)),
        "num_negative": int(np.sum(y_true == 0)),
        "auc": auc,
        "roc_points": points,
        "scored_rows_preview": rows[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute ROC curve points and AUC from score/key files.")
    parser.add_argument("--score", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--phase", default="eval")
    parser.add_argument("--positive-label", default="bonafide", choices=["bonafide", "spoof"])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-svg", default=None)
    args = parser.parse_args()

    result = compute_roc_auc(
        args.score,
        args.metadata,
        phase=args.phase,
        positive_label=args.positive_label,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.output_csv:
        write_csv(result["roc_points"], Path(args.output_csv))
    if args.output_svg:
        write_svg(result["roc_points"], Path(args.output_svg))

    print(json.dumps({k: v for k, v in result.items() if k != "roc_points"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
