import argparse
import json
from pathlib import Path

import numpy as np


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


class ScoreCalibrator:
    """Percentile-based score calibration for the joint decision matrix."""

    def __init__(self, voice_sim_scores=None, fake_prob_scores=None):
        self.voice_percentiles = (
            np.percentile(np.asarray(voice_sim_scores, dtype=float), np.arange(0, 101, 1))
            if voice_sim_scores is not None and len(voice_sim_scores) > 0
            else None
        )
        self.fake_percentiles = (
            np.percentile(np.asarray(fake_prob_scores, dtype=float), np.arange(0, 101, 1))
            if fake_prob_scores is not None and len(fake_prob_scores) > 0
            else None
        )

    def calibrate_voice(self, raw_score):
        if raw_score is None:
            return None
        raw_score = float(raw_score)
        if self.voice_percentiles is None:
            # Cosine similarity usually falls in [-1, 1]. Map it to [0, 1]
            # when no validation distribution is available.
            return clamp01((raw_score + 1.0) / 2.0)
        return clamp01(np.searchsorted(self.voice_percentiles, raw_score) / 100.0)

    def calibrate_fake(self, raw_score):
        if raw_score is None:
            return None
        raw_score = float(raw_score)
        if self.fake_percentiles is None:
            return clamp01(raw_score)
        return clamp01(np.searchsorted(self.fake_percentiles, raw_score) / 100.0)


class JointRiskAssessment:
    def __init__(self, calibrator=None, voice_threshold=0.70, fake_threshold=0.50):
        self.calibrator = calibrator or ScoreCalibrator()
        self.voice_threshold = float(voice_threshold)
        self.fake_threshold = float(fake_threshold)

    def assess(self, raw_voice_similarity=None, raw_fake_probability=None):
        fake_score = self.calibrator.calibrate_fake(raw_fake_probability)
        voice_score = self.calibrator.calibrate_voice(raw_voice_similarity)

        if fake_score is None:
            return {
                "available": False,
                "reason": "fake probability is required",
            }

        if voice_score is None:
            high_fake = fake_score >= self.fake_threshold
            return {
                "available": False,
                "reason": "voice similarity is not available",
                "fake_only_level": "high" if high_fake else "low",
                "calibrated_fake_score": round(fake_score, 3),
                "fake_threshold": self.fake_threshold,
            }

        high_voice = voice_score >= self.voice_threshold
        high_fake = fake_score >= self.fake_threshold

        if high_voice and high_fake:
            level, desc, quadrant = (
                "高危",
                "声纹高度匹配目标人物，且检测为 AI 合成语音，疑似精准仿冒攻击，建议立即预警并人工复核。",
                "Q2",
            )
        elif high_voice and not high_fake:
            level, desc, quadrant = (
                "低风险",
                "声纹匹配目标人物，且未检测到明显伪造特征，疑似本人真实发言。",
                "Q1",
            )
        elif not high_voice and high_fake:
            level, desc, quadrant = (
                "中危",
                "未匹配到目标人物声纹，但检测为 AI 合成语音，可能为冒用其他身份或仿真度有限的伪造内容。",
                "Q4",
            )
        else:
            level, desc, quadrant = (
                "低风险",
                "未匹配目标人物声纹，且未检测到明显伪造特征。",
                "Q3",
            )

        return {
            "available": True,
            "risk_level": level,
            "risk_description": desc,
            "quadrant": quadrant,
            "raw_voice_similarity": None if raw_voice_similarity is None else float(raw_voice_similarity),
            "raw_fake_probability": None if raw_fake_probability is None else float(raw_fake_probability),
            "calibrated_voice_score": round(voice_score, 3),
            "calibrated_fake_score": round(fake_score, 3),
            "voice_threshold": self.voice_threshold,
            "fake_threshold": self.fake_threshold,
        }


def assess_joint_risk(
    voice_similarity=None,
    fake_probability=None,
    voice_threshold=0.70,
    fake_threshold=0.50,
):
    assessor = JointRiskAssessment(
        voice_threshold=voice_threshold,
        fake_threshold=fake_threshold,
    )
    return assessor.assess(
        raw_voice_similarity=voice_similarity,
        raw_fake_probability=fake_probability,
    )


def main():
    parser = argparse.ArgumentParser(description="Assess voice similarity and fake probability jointly.")
    parser.add_argument("--voice-similarity", type=float)
    parser.add_argument("--fake-probability", type=float, required=True)
    parser.add_argument("--voice-threshold", type=float, default=0.70)
    parser.add_argument("--fake-threshold", type=float, default=0.50)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = assess_joint_risk(
        voice_similarity=args.voice_similarity,
        fake_probability=args.fake_probability,
        voice_threshold=args.voice_threshold,
        fake_threshold=args.fake_threshold,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
