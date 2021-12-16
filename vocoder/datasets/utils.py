from operator import xor

from torch.utils.data import DataLoader, ConcatDataset

import tts.datasets
from ..collator import LJSpeechCollator
from ..utils import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, tts.datasets, split=split, config_parser=configs
            ))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            batch_size = params["batch_size"]
            shuffle = True
            batch_sampler = None
        else:
            raise Exception()

        # create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LJSpeechCollator(), shuffle=True)
        dataloaders[split] = dataloader
    return dataloaders
