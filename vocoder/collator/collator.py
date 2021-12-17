import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
from .utils import MelSpectrogram, MelSpectrogramConfig


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    melspec_real: Optional[torch.Tensor] = None
    waveform_gen: Optional[torch.Tensor] = None
    melspec_gen: Optional[torch.Tensor] = None
    # MPD
    mpd_gen: Optional[List[torch.Tensor]] = None
    mpd_real: Optional[List[torch.Tensor]] = None
    mpd_feat_gen: Optional[List[List[torch.Tensor]]] = None
    mpd_feat_real: Optional[List[List[torch.Tensor]]] = None
    # MSD
    msd_gen: Optional[List[torch.Tensor]] = None
    msd_real: Optional[List[torch.Tensor]] = None
    msd_feat_gen: Optional[List[List[torch.Tensor]]] = None
    msd_feat_real: Optional[List[List[torch.Tensor]]] = None

    def to(self, device: torch.device) -> 'Batch':
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                value = value.to(device)
                self.__setattr__(key, value)
        return self

    def __getitem__(self, key):
        return self.__dict__[key]


class LJSpeechCollator:
    def __init__(self):
        self.melspec_config = MelSpectrogramConfig()
        self.melspec_silence = -11.5129251

    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length, transcript = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        return Batch(
            waveform, waveform_length, transcript
        )
