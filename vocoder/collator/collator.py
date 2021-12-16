import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
from .utils import MelSpectrogram, MelSpectrogramConfig
from ..aligner import GraphemeAligner


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    melspec_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None
    melspec: Optional[torch.Tensor] = None
    melspec_pred: Optional[torch.Tensor] = None
    durations_pred: Optional[torch.Tensor] = None
    phoneme: Optional[torch.Tensor] = None

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
        # self.aligner = GraphemeAligner()
        self.melspec_silence = -11.5129251

    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveforn_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        # melspec = [
        #     self.melspec(waveform_[0]) for waveform_ in waveform
        # ]
        # melspec_length = torch.Tensor([melspec_.size(-1) for melspec_ in melspec])
        # melspec = pad_sequence([
        #     melspec_.transpose(1, 0) for melspec_ in melspec
        # ], padding_value=self.melspec_silence)\
        #     .transpose(1, 0).transpose(2, 1)

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveforn_length)

        melspec_length = (waveform_length - self.melspec_config.win_length) // self.melspec_config.hop_length + 5

        # durations = self.aligner(
        #     waveform, waveform_length, transcript
        # )

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(
            waveform, waveform_length, melspec_length,
            # melspec, melspec_length, durations,
            transcript, tokens, token_lengths
        )
