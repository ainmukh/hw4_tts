import torch
import torch.nn as nn
import torchaudio
from dataclasses import dataclass
import librosa


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):
    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        # audio, self.n_fft, self.num_mels,
        # self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
        # center = False)

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


# def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
#     if torch.min(y) < -1.:
#         print('min value is ', torch.min(y))
#     if torch.max(y) > 1.:
#         print('max value is ', torch.max(y))
#
#     global mel_basis, hann_window
#     if fmax not in mel_basis:
#         mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
#         mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
#         hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
#
#     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
#     y = y.squeeze(1)
#
#     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
#                       center=center, pad_mode='reflect', normalized=False, onesided=True)
#
#     spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
#
#     spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
#     spec = spectral_normalize_torch(spec)
#
#     return spec