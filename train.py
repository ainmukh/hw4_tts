import argparse
import collections
import warnings
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

import vocoder.loss as module_loss
import vocoder.model as module_arch
from vocoder.datasets import LJSpeechDataset, get_dataloaders
from vocoder.trainer import Trainer
from vocoder.utils import prepare_device, ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    batch_size = config['data']['train']['batch_size']
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    generator = config.init_obj(config['arch']['generator'], module_arch)
    print('got generator')
    mpd = config.init_obj(config['arch']['mpd'], module_arch)
    print('got mpd')
    msd = config.init_obj(config['arch']['msd'], module_arch)
    print('got msd')

    if config['warm_start'] != '':
        print('Starting from checkpoint', config['warm_start'])
        check_point = torch.load(config['warm_start'])
        generator.load_state_dict(check_point['gen_state_dict'])
        mpd.load_state_dict(check_point['mpd_state_dict'])
        msd.load_state_dict(check_point['msd_state_dict'])
    logger.info(generator)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    generator = generator.to(device)
    mpd, msd = mpd.to(device), msd.to(device)

    # get function handles of loss and metrics
    gen_loss_module = config.init_obj(config['loss']['gen_loss'], module_loss).to(device)
    dis_loss_module = config.init_obj(config['loss']['dis_loss'], module_loss).to(device)
    metrics = []

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    gen_optimizer = config.init_obj(config['optimizer']['gen_optimizer'], torch.optim, trainable_params)
    trainable_params = filter(lambda p: p.requires_grad, itertools.chain(mpd.parameters(), msd.parameters()))
    dis_optimizer = config.init_obj(config['optimizer']['dis_optimizer'], torch.optim, trainable_params)

    # lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        generator, mpd, msd,
        gen_loss_module, dis_loss_module,
        gen_optimizer, dis_optimizer,
        config,
        device,
        dataloaders['train'],
        valid_data_loader=dataloaders['val'],
        len_epoch=config['trainer'].get('len_epoch', None),
        sr=config['preprocessing']['sr']
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
