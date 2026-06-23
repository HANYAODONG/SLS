FORBIDDEN_PHRASES = [
    "确定由",
    "已经证明",
    "百分之百",
    "100%",
    "必然",
]


def validate_report(report):
    errors = []
    for phrase in FORBIDDEN_PHRASES:
        if phrase in report:
            errors.append("forbidden phrase: {}".format(phrase))
    required = ["检测结论", "关键证据", "稳定性", "限制说明"]
    for title in required:
        if title not in report:
            errors.append("missing section: {}".format(title))
    return {"ok": not errors, "errors": errors}
