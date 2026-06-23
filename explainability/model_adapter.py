import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from explainability.audio_utils import save_audio_16k
from model import Model, getAttenF
from single_audio_infer import load_processed_audio, pad_audio, run_ffmpeg_convert


class SLSModelAdapter:
    """Read-only adapter around the original SLS model.

    The adapter does not modify model.py or add trainable parameters. It only
    exposes prediction and SLS layer-weight details needed by post-hoc
    explainability routines.
    """

    def __init__(self, checkpoint, xlsr_checkpoint="xlsr2_300m.pt", device=None, disable_cudnn=True):
        if disable_cudnn:
            torch.backends.cudnn.enabled = False
        self.checkpoint = str(checkpoint)
        self.xlsr_checkpoint = str(xlsr_checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        args = SimpleNamespace(xlsr_checkpoint=self.xlsr_checkpoint)
        model = Model(args, self.device)
        self.model = nn.DataParallel(model).to(self.device)
        state_dict = torch.load(self.checkpoint, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        del state_dict
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def preprocess_file(self, audio_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            converted = Path(tmpdir) / "audio_16k.wav"
            run_ffmpeg_convert(audio_path, converted)
            audio, _ = load_processed_audio(converted, sr=16000)
        return audio

    def tensor_from_audio(self, audio):
        x_pad = pad_audio(np.asarray(audio, dtype=np.float32), 64600)
        return torch.tensor(x_pad, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict_audio(self, audio):
        batch_x = self.tensor_from_audio(audio)
        with torch.no_grad():
            log_probs = self.model(batch_x)
            probs = torch.exp(log_probs)
        return {
            "log_probabilities": log_probs.detach().cpu().numpy().ravel().tolist(),
            "probabilities": probs.detach().cpu().numpy().ravel().tolist(),
            "fake_probability": float(probs.detach().cpu().numpy().ravel()[0]),
            "bonafide_probability": float(probs.detach().cpu().numpy().ravel()[1]),
        }

    def predict_file(self, audio_path):
        return self.predict_audio(self.preprocess_file(audio_path))

    def details_audio(self, audio):
        batch_x = self.tensor_from_audio(audio)
        module = self.model.module
        with torch.no_grad():
            _, layer_result = module.ssl_model.extract_feat(batch_x.squeeze(-1))
            pooled_layers, full_features = getAttenF(layer_result)
            layer_weights = module.sig(module.fc0(pooled_layers)).squeeze(-1).squeeze(-1)
            fused_sequence = torch.sum(full_features * layer_weights.view(layer_weights.shape[0], -1, 1, 1), dim=1)
            prediction = self.model(batch_x)
            probs = torch.exp(prediction)
        return {
            "hidden_states_shape": list(full_features.shape),
            "fused_sequence_shape": list(fused_sequence.shape),
            "layer_weights": layer_weights.detach().cpu().numpy().ravel().tolist(),
            "log_probabilities": prediction.detach().cpu().numpy().ravel().tolist(),
            "probabilities": probs.detach().cpu().numpy().ravel().tolist(),
            "fake_probability": float(probs.detach().cpu().numpy().ravel()[0]),
            "bonafide_probability": float(probs.detach().cpu().numpy().ravel()[1]),
        }

    def predict_temp_wav(self, audio):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "variant.wav"
            save_audio_16k(audio, path)
            return self.predict_file(path)
