import json
import logging
import os
import shutil

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from tts.base import BaseDataset
from tts.utils import ROOT_PATH

import pandas as pd


logger = logging.getLogger(__name__)

URL_LINKS = ['https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2']


class LJSpeechBase(BaseDataset):
    def __init__(self, data_dir=None, split=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self.split = split
        self._data_dir = data_dir
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _load_part(self):
        pass
        arch_path = self._data_dir / f"LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS[0], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def _get_or_load_index(self):
        index_path = self._data_dir / f"lj_index_{self.split}.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        split_dir = self._data_dir / 'LJSpeech-1.1.tar.bz2'
        # print('split_dir', split_dir, split_dir.exists(), self.split)
        if not split_dir.exists() and self.split == "train":
            self._load_part()

        pass_token = '<pass>'
        df_path = self._data_dir / "metadata.csv"
        df = pd.read_csv(df_path, sep='|', header=None)
        df.fillna(pass_token, inplace=True)
        df = df[:11000] if self.split == "train" else df[11000:]
        wav_dir = self._data_dir / 'wavs'
        for wav_id in tqdm(df[0]):
            if df[df[0] == wav_id][2].values[0] == pass_token:
                continue
            if not df[df[0] == wav_id][2].values[0].isascii():
                continue
            if '[' in df[df[0] == wav_id][2].values[0]:
                continue
            wav_text = df[df[0] == wav_id][2].values[0]
            wav_path = wav_dir / f"{wav_id}.wav"
            t_info = torchaudio.info(str(wav_path))
            length = t_info.num_frames / t_info.sample_rate
            index.append(
                {
                    "path": str(wav_path.absolute().resolve()),
                    "text": wav_text,
                    "audio_len": length,
                }
            )
        return index
