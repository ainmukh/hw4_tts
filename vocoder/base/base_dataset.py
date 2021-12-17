import logging
import random
import re

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import ConfigParser


logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            limit=-1,
            max_audio_length=None,
            max_text_length=None,
            segment_size=None
    ):
        self.config_parser = config_parser

        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )

        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length, limit
        )

        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index = index
        self.segment_size = segment_size

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave, sample_rate = torchaudio.load(audio_path)

        # print('audio_wave size =', audio_wave.size())
        if self.segment_size is not None and audio_wave.size(-1) >= self.segment_size:
            max_audio_start = audio_wave.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio_wave = audio_wave[:, audio_start:audio_start + self.segment_size]

        text = data_dict['text']
        return audio_wave, sample_rate, text

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, max_text_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array(
                [1 if el["audio_len"] >= max_audio_length else 0 for el in index]
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_text_length = np.array(
                [
                    1 if len(el["text"]) >= max_text_length else 0
                    for el in index
                ]
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        # if records_to_filter is not False and records_to_filter.any():
        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records  from dataset"
            )

        if limit > 0:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        print('records remain:', len(index))
        return index
