import torch
import torchaudio
from ..base import LJSpeechBase
from ..utils import ConfigParser


class LJSpeechDataset(LJSpeechBase):

    def __init__(self, data_dir=None, split=None, *args, **kwargs):
        super().__init__(data_dir=data_dir, split=split, *args, **kwargs)

    def __getitem__(self, index: int):
        waveform, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        return waveform, waveform_length, transcript


if __name__ == "__main__":
    config_parser = ConfigParser.get_default_configs()

    ds = LJSpeechDataset(
        config_parser=config_parser
    )
    item = ds[0]
    print(item)
