import torch
import torchaudio
from ..base import LJSpeechBase
from ..utils import ConfigParser


class LJSpeechDataset(LJSpeechBase):

    def __init__(self, data_dir=None, split=None, *args, **kwargs):
        super().__init__(data_dir=data_dir, split=split, *args, **kwargs)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveforn_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


if __name__ == "__main__":
    config_parser = ConfigParser.get_default_configs()

    ds = LJSpeechDataset(
        config_parser=config_parser
    )
    item = ds[0]
    print(item)
