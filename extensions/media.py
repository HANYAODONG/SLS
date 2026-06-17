import json
import shutil
import subprocess
from pathlib import Path


def resolve_command(command):
    path = shutil.which(command)
    if path:
        return path
    if command == "ffmpeg":
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH, and imageio-ffmpeg is unavailable."
            ) from exc
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise RuntimeError(
        "{} is not installed or not in PATH. Install it before using this extension."
        .format(command)
    )


def run_command(args):
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def extract_audio(video_path, output_audio, sample_rate=16000):
    ffmpeg = resolve_command("ffmpeg")
    output_audio = Path(output_audio)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(output_audio),
        ]
    )
    return output_audio


def extract_keyframes(video_path, output_dir, fps=1.0):
    ffmpeg = resolve_command("ffmpeg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = output_dir / "frame_%05d.jpg"
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "fps={}".format(fps),
            "-q:v",
            "2",
            str(frame_pattern),
        ]
    )
    return sorted(output_dir.glob("frame_*.jpg"))


def probe_media(media_path):
    try:
        ffprobe = resolve_command("ffprobe")
    except RuntimeError as exc:
        return {
            "available": False,
            "error": str(exc),
            "note": "Install system ffprobe for detailed media metadata.",
        }
    output = run_command(
        [
            ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(media_path),
        ]
    )
    payload = json.loads(output)
    payload["available"] = True
    return payload
