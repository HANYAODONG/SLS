import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_metrics_DF as df_metrics


def read_scores(score_path):
    scores = {}
    with open(score_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError("Invalid score line: {}".format(line.rstrip()))
            scores[parts[0]] = float(parts[1])
    return scores


def read_metadata(metadata_path):
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    return rows


def describe(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "count": count,
        "mean": mean,
        "std": variance ** 0.5,
        "min": min(values),
        "max": max(values),
    }


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def compute_df_stats(score_path, metadata_path, phase="eval", experiment=None):
    score_path = Path(score_path)
    metadata_path = Path(metadata_path)
    scores = read_scores(score_path)
    metadata_rows = read_metadata(metadata_path)

    phase_rows = [row for row in metadata_rows if len(row) > 7 and row[7] == phase]
    phase_scores = []
    bonafide_scores = []
    spoof_scores = []
    missing_score_ids = []

    for row in phase_rows:
        utt_id = row[1]
        label = row[5]
        if utt_id not in scores:
            missing_score_ids.append(utt_id)
            continue
        score = scores[utt_id]
        phase_scores.append(score)
        if label == "bonafide":
            bonafide_scores.append(score)
        elif label == "spoof":
            spoof_scores.append(score)

    eer = None
    if bonafide_scores and spoof_scores:
        eer = 100 * df_metrics.compute_eer(np.asarray(bonafide_scores), np.asarray(spoof_scores))[0]

    all_score_values = list(scores.values())
    score_distribution = describe(all_score_values)
    score_distribution.update(
        {
            "p25": percentile(all_score_values, 0.25),
            "p50": percentile(all_score_values, 0.50),
            "p75": percentile(all_score_values, 0.75),
        }
    )

    return {
        "experiment": experiment or score_path.stem,
        "score_file": str(score_path),
        "metadata_file": str(metadata_path),
        "phase": phase,
        "num_scores": len(scores),
        "num_metadata_rows": len(metadata_rows),
        "num_phase_rows": len(phase_rows),
        "num_scored_phase_rows": len(phase_scores),
        "unique_score_ids": len(scores),
        "phase_counts": dict(Counter(row[7] for row in metadata_rows if len(row) > 7)),
        "label_counts_total": dict(Counter(row[5] for row in metadata_rows if len(row) > 5)),
        "label_counts_phase": dict(Counter(row[5] for row in phase_rows if len(row) > 5)),
        "missing_score_count": len(missing_score_ids),
        "missing_score_examples": missing_score_ids[:10],
        "eer_percent": eer,
        "score_distribution": score_distribution,
        "bonafide_score_distribution": describe(bonafide_scores),
        "spoof_score_distribution": describe(spoof_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute local DF score statistics.")
    parser.add_argument("--score", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--phase", default="eval")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    stats = compute_df_stats(args.score, args.metadata, args.phase, args.experiment)
    payload = json.dumps(stats, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
