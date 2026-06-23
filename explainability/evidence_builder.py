from pathlib import Path


def top_items(items, limit):
    return list(items[: int(limit)])


def build_evidence(
    audio_path,
    audio,
    prediction,
    details,
    time_rows,
    frequency_rows,
    stability,
    config,
):
    sample_rate = int(config.get("sample_rate", 16000))
    layer_weights = details.get("layer_weights", [])
    top_layers = sorted(
        [
            {"layer_number": index, "sls_weight": float(weight)}
            for index, weight in enumerate(layer_weights)
        ],
        key=lambda item: item["sls_weight"],
        reverse=True,
    )
    fake_probability = prediction["fake_probability"]
    label = "fake" if fake_probability >= 0.5 else "bonafide"

    return {
        "schema_version": "1.0",
        "audio": {
            "path": str(audio_path),
            "name": Path(audio_path).name,
            "sample_rate": sample_rate,
            "duration_seconds": len(audio) / float(sample_rate),
        },
        "decision": {
            "label": label,
            "fake_probability": fake_probability,
            "bonafide_probability": prediction["bonafide_probability"],
            "raw_logits": prediction["log_probabilities"],
            "fake_class_index": int(config.get("fake_class_index", 0)),
        },
        "model_details": {
            "hidden_states_shape": details.get("hidden_states_shape"),
            "fused_sequence_shape": details.get("fused_sequence_shape"),
        },
        "layer_evidence": {
            "top_sls_layers": top_items(top_layers, config.get("top_layer_ablation", 6)),
            "layer_ablation": [],
            "layer_ablation_note": "当前实现不修改 model.py，未启用内部层遮挡；保留 SLS 层权重作为解释证据。",
        },
        "temporal_evidence": {
            "top_occlusion_segments": top_items(time_rows, config.get("top_segments", 5)),
        },
        "frequency_evidence": {
            "top_frequency_bands": top_items(frequency_rows, config.get("top_frequency_bands", 3)),
        },
        "stability": stability,
        "prototype_and_ood": None,
        "report_constraints": {
            "must_not_claim_generator_identity": True,
            "must_not_claim_causality": True,
            "must_include_ood_warning": False,
            "must_include_stability": True,
            "must_state_detection_limitations": True,
        },
    }
