import argparse
import collections
import warnings

import PIL
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import numpy as np
import pandas as pd
import wandb

import vocoder.model as module_arch
from vocoder.utils import prepare_device, ConfigParser, ROOT_PATH
from vocoder.collator import MelSpectrogramConfig, MelSpectrogram
from vocoder.logger import plot_spectrogram_to_buf

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / 'config1.json'
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / 'saved/model_best.pth'


def main(config):
    logger = config.get_logger('test')

    # build model architecture
    generator = config.init_obj(config['arch']['generator'], module_arch)
    logger.info(generator)

    device, device_ids = prepare_device(config['n_gpu'])
    generator = generator.to(device)
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['gen_state_dict']
    generator.load_state_dict(state_dict)

    # prepare model for testing
    generator.eval()

    # get data
    data = pd.read_csv(ROOT_PATH / 'data/test/postmeta.csv')
    batch = {
        'transcript': [],
        'wave_form': []
    }
    for i in range(data.shape[0]):
        _, transcript, audio_path = data.iloc[i]
        audio_path = ROOT_PATH / 'data/test' / audio_path
        audio_wave, sample_rate = torchaudio.load(audio_path)
        batch['wave_form'].append(audio_wave)
        batch['transcript'].append(transcript)
    batch['sample_rate'] = sample_rate

    # get batch
    waveform = pad_sequence([
        waveform_[0] for waveform_ in batch['wave_form']
    ]).transpose(0, 1)
    batch['wave_form'] = waveform.to(device)
    batch['wave_form'].size()

    batch['melspec_real'] = featurizer(batch['wave_form'])
    batch['waveform_gen'] = generator(batch['melspec_real'])
    batch['melspec_gen'] = featurizer(batch['waveform_gen'].squeeze(1))

    # log
    project_name = config['trainer']['project_name']
    wandb.init(project=project_name)
    for i, wave in enumerate(batch['waveform_gen']):
        audio = wandb.Audio(
            wave.squeeze(0).cpu().detach().numpy(),
            caption=batch['transcript'][i],
            sample_rate=batch['sample_rate']
        )
        melspec_real = PIL.Image.open(plot_spectrogram_to_buf(batch['melspec_real'][i].detach().cpu()))
        melspec_real = wandb.Image(
            # batch['melspec_real'][i].cpu().detach().numpy(),
            melspec_real,
            caption=batch['transcript'][i]
        )
        melspec_gen = wandb.Image(
            batch['melspec_gen'][i].cpu().detach().numpy(),
            caption=batch['transcript'][i]
        )
        wandb.log({
            'audio': audio,
            'melspec_real': melspec_real,
            'melspec_gen': melspec_gen
        }, step=i + 1)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_TEST_CONFIG_PATH.absolute().resolve()),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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
    # args.add_argument(
    #     "-t",
    #     "--test-data-folder",
    #     default=None,
    #     required=True,
    #     type=str,
    #     help="Path to dataset",
    # )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader"
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)
