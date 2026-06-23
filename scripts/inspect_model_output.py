import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.model_adapter import SLSModelAdapter


def main():
    parser = argparse.ArgumentParser(description="Inspect original SLS model details without changing model.py.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", default="MMpaper_model.pth")
    parser.add_argument("--xlsr-checkpoint", default="xlsr2_300m.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    adapter = SLSModelAdapter(args.checkpoint, xlsr_checkpoint=args.xlsr_checkpoint, device=args.device)
    audio = adapter.preprocess_file(args.audio)
    details = adapter.details_audio(audio)
    payload = {
        "audio_samples": int(audio.shape[0]),
        "fixed_input_length": 64600,
        "fake_class_index": 0,
        "details": {
            "hidden_states_shape": details["hidden_states_shape"],
            "fused_sequence_shape": details["fused_sequence_shape"],
            "layer_weights_shape": [len(details["layer_weights"])],
            "log_probabilities": details["log_probabilities"],
            "probabilities": details["probabilities"],
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
