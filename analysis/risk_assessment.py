import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_metrics_DF as df_metrics
from analysis.score_stats import read_metadata, read_scores, row_label, row_utt_id


def compute_threshold(scores, metadata_rows):
    bonafide_scores = []
    spoof_scores = []
    for row in metadata_rows:
        utt_id = row_utt_id(row)
        if utt_id not in scores:
            continue
        label = row_label(row)
        if label == "bonafide":
            bonafide_scores.append(scores[utt_id])
        elif label == "spoof":
            spoof_scores.append(scores[utt_id])

    if not bonafide_scores or not spoof_scores:
        return None
    return float(
        df_metrics.compute_eer(
            np.asarray(bonafide_scores),
            np.asarray(spoof_scores),
        )[1]
    )


def risk_level(score, threshold):
    if threshold is None:
        return "unknown"
    if score < threshold:
        return "high"
    return "low"


def build_risk_rows(score_path, metadata_path, top_n=50):
    scores = read_scores(score_path)
    metadata_rows = read_metadata(metadata_path)
    threshold = compute_threshold(scores, metadata_rows)
    rows = []

    for row in metadata_rows:
        utt_id = row_utt_id(row)
        if utt_id not in scores:
            continue
        score = scores[utt_id]
        label = row_label(row)
        predicted = "spoof" if threshold is not None and score < threshold else "bonafide"
        rows.append(
            {
                "utt_id": utt_id,
                "score": score,
                "label": label,
                "predicted": predicted,
                "risk_level": risk_level(score, threshold),
                "threshold": threshold,
                "distance_to_threshold": None if threshold is None else abs(score - threshold),
            }
        )

    rows.sort(key=lambda item: item["score"])
    return rows[:top_n]


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "utt_id",
        "score",
        "label",
        "predicted",
        "risk_level",
        "threshold",
        "distance_to_threshold",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Export high-risk spoof-like score rows.")
    parser.add_argument("--score", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_risk_rows(args.score, args.metadata, top_n=args.top_n)
    write_csv(rows, args.output)
    print("Risk list saved to {}".format(args.output))


if __name__ == "__main__":
    main()
