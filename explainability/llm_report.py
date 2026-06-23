def fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return ("{:.%df}" % digits).format(value)
    return str(value)


def template_report(evidence):
    decision = evidence["decision"]
    temporal = evidence["temporal_evidence"]["top_occlusion_segments"]
    frequency = evidence["frequency_evidence"]["top_frequency_bands"]
    layers = evidence["layer_evidence"]["top_sls_layers"]
    stability = evidence["stability"]

    label_text = "疑似伪造" if decision["label"] == "fake" else "疑似真实"
    lines = [
        "检测结论：",
        "当前模型将该音频判定为{}，模型伪造类别输出值为{}。".format(
            label_text,
            fmt(decision["fake_probability"]),
        ),
        "",
        "关键证据：",
    ]

    if temporal:
        top = temporal[0]
        lines.append(
            "时间遮挡中，遮挡 {:.2f} 至 {:.2f} 秒后，伪造输出平均变化为 {}。".format(
                top["start"],
                top["end"],
                fmt(top["average_probability_drop"]),
            )
        )
    if frequency:
        top = frequency[0]
        lines.append(
            "频带遮挡中，平滑衰减 {:.0f} 至 {:.0f} Hz 后，伪造输出变化为 {}。".format(
                top["low_hz"],
                top["high_hz"],
                fmt(top["probability_drop"]),
            )
        )
    if layers:
        top_layers = ", ".join(
            "第{}层({})".format(item["layer_number"], fmt(item["sls_weight"], 3))
            for item in layers[:3]
        )
        lines.append("SLS 层权重较高的层包括：{}。".format(top_layers))

    lines.extend(
        [
            "",
            "稳定性：",
            "扰动测试下分数波动范围为 {}，稳定性等级为 {}。".format(
                fmt(stability.get("range")),
                stability.get("level", "N/A"),
            ),
            "",
            "风险提示：",
            "以上证据描述的是模型输出在遮挡和扰动后的变化，用于辅助理解模型判别依据。",
            "",
            "限制说明：",
            "该报告不能证明音频的具体生成器来源，不能把关注片段解释为已证实伪造位置，也不能替代人工复核或司法鉴定。",
        ]
    )
    return "\n".join(lines)
