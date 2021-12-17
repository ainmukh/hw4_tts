import torch
from torch.nn import L1Loss


class L1LossMelSpec(L1Loss):
    def forward(self, mel_gen, mel_real):
        """
        MelSpectrogram loss
        :param mel_gen: mel spectrogram from generated wav
        :param mel_real: mel spectrogram from real wav
        :return: loss
        """
        return super(L1LossMelSpec, self).forward(mel_gen, mel_real)
