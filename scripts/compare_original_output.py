import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.model_adapter import SLSModelAdapter


def main():
    parser = argparse.ArgumentParser(description="Compare repeated original SLS outputs through the adapter.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", default="MMpaper_model.pth")
    parser.add_argument("--xlsr-checkpoint", default="xlsr2_300m.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    adapter = SLSModelAdapter(args.checkpoint, xlsr_checkpoint=args.xlsr_checkpoint, device=args.device)
    audio = adapter.preprocess_file(args.audio)
    output_a = adapter.predict_audio(audio)["log_probabilities"]
    output_b = adapter.details_audio(audio)["log_probabilities"]
    difference = float(np.max(np.abs(np.asarray(output_a) - np.asarray(output_b))))
    print(
        json.dumps(
            {
                "max_abs_difference": difference,
                "output_a": output_a,
                "output_b": output_b,
                "ok_cpu_tolerance_1e-7": difference < 1e-7,
                "ok_gpu_tolerance_1e-5": difference < 1e-5,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
