import random
import json

import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..base import BaseTrainer
from ..logger import plot_spectrogram_to_buf
from ..utils import inf_loop, MetricTracker
from ..collator import MelSpectrogram, MelSpectrogramConfig

from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator, mpd, msd,
            gen_criterion, dis_criterion,
            gen_optimizer, dis_optimizer,
            config,
            device,
            data_loader,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            sr=22050
    ):
        super().__init__(
            generator, mpd, msd,
            gen_criterion, dis_criterion,
            gen_optimizer, dis_optimizer,
            config, device
        )
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader

        self.melspec = MelSpectrogram(MelSpectrogramConfig()).to(device)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 10

        self.train_metrics = MetricTracker(
            'mel_loss', 'fm_loss', 'gen_loss', 'grad norm', writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            'mel_loss', 'fm_loss', 'gen_loss', writer=self.writer
        )
        self.sr = sr
        self.overfit = False

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_iteration(self, batch, epoch: int, batch_num: int):
        batch = batch.to(self.device)

        # batch.melspec_real = self.melspec(batch.waveform)
        batch.melspec_real = mel_spectrogram(
            batch.waveform,
            n_fft=1024, num_mels=80,
            sampling_rate=22050, hop_size=256,
            win_size=1024, fmin=0, fmax=8000, center=False
        )
        batch.waveform = batch.waveform.unsqueeze(1)
        # print('mel, audio size =', batch.melspec_real.size(), batch.waveform.size())
        batch.waveform_gen = self.generator(batch.melspec_real)
        # print('gen waveform size =', batch.waveform_gen.size())
        melspec_gen = self.melspec(batch.waveform_gen.squeeze(1))
        # print('my melspec =', melspec_gen.size())
        batch.melspec_gen = mel_spectrogram(
            batch.waveform_gen.squeeze(1),
            n_fft=1024, num_mels=80,
            sampling_rate=22050, hop_size=256,
            win_size=1024, fmin=0, fmax=8000, center=False
        )

        # DISCRIMINATOR
        if not self.overfit:
            self.optimizer_dis.zero_grad()
            # MPD
            mpd_real, mpd_gen, _, _ = self.mpd(batch.waveform, batch.waveform_gen.detach())
            mpd_loss = self.dis_criterion(mpd_gen, mpd_real)
            # MSD
            msd_real, msd_gen, _, _ = self.msd(batch.waveform, batch.waveform_gen.detach())
            msd_loss = self.dis_criterion(msd_gen, msd_real)

            discriminator_loss = mpd_loss + msd_loss
            discriminator_loss.backward()
            self.optimizer_dis.step()

        # GENERATOR
        self.optimizer_gen.zero_grad()
        if not self.overfit:
            batch.mpd_real, batch.mpd_gen, \
                batch.mpd_feat_real, batch.mpd_feat_gen = self.mpd(
                    batch.waveform, batch.waveform_gen
                )
            batch.msd_real, batch.msd_gen, \
                batch.msd_feat_real, batch.msd_feat_gen = self.msd(
                    batch.waveform, batch.waveform_gen
                )
        generator_loss, fm_loss, mel_loss = self.gen_criterion(batch)
        generator_loss.backward()
        self.optimizer_gen.step()

        self.writer.set_step((epoch - 1) * self.len_epoch + batch_num)
        self.train_metrics.update('mel_loss', mel_loss.item())
        if fm_loss is not None:
            self.train_metrics.update('fm_loss', fm_loss.item())
        self.train_metrics.update('gen_loss', generator_loss.item())
        self.train_metrics.update('grad norm', self.get_grad_norm())

        if batch_num % self.log_step == 0 and batch_num:
            # self.writer.add_scalar(
            #     "learning rate", self.lr_scheduler.get_last_lr()[0]
            # )
            self._log_predictions(batch.melspec_gen, batch.transcript)
            self._log_scalars(self.train_metrics)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        if not self.overfit:
            self.mpd.train()
            self.msd.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            try:
                self._train_iteration(batch, epoch, batch_idx)
            except RuntimeError as e:
                if 'out of memory' in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.generator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx >= self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        if not self.overfit:
            self.mpd.eval()
            self.msd.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader), desc="validation",
                    total=len(self.valid_data_loader)
            ):
                batch = batch.to(self.device)

                # batch.melspec_real = self.melspec(batch.waveform)
                batch.melspec_real = mel_spectrogram(
                    batch.waveform,
                    n_fft=1024, num_mels=80,
                    sampling_rate=22050, hop_size=256,
                    win_size=1024, fmin=0, fmax=8000, center=False
                )
                batch.waveform = batch.waveform.unsqueeze(1)
                # print('mel, audio size =', batch.melspec_real.size(), batch.waveform.size())
                batch.waveform_gen = self.generator(batch.melspec_real)
                # print('gen waveform size =', batch.waveform_gen.size())
                # batch.melspec_gen = self.melspec(batch.waveform_gen.squeeze(0))
                batch.melspec_gen = mel_spectrogram(
                    batch.waveform_gen.squeeze(1),
                    n_fft=1024, num_mels=80,
                    sampling_rate=22050, hop_size=256,
                    win_size=1024, fmin=0, fmax=8000, center=False
                )

                if not self.overfit:
                    batch.mpd_real, batch.mpd_gen, batch.mpd_feat_real, batch.mpd_feat_gen = self.mpd(
                            batch.waveform, batch.waveform_gen
                    )
                    batch.msd_real, batch.msd_gen, batch.msd_feat_real, batch.msd_feat_gen = self.msd(
                            batch.waveform, batch.waveform_gen
                    )
                generator_loss, fm_loss, mel_loss = self.gen_criterion(batch)

                self.valid_metrics.update('mel_loss', mel_loss.item(), n=batch.melspec_gen.size(0))
                if fm_loss is not None:
                    self.train_metrics.update('fm_loss', fm_loss.item(), n=batch.melspec_gen.size(0))
                self.valid_metrics.update('gen_loss', generator_loss.item(), n=batch.melspec_gen.size(0))

            self.writer.set_step(epoch * self.len_epoch, "valid")
            self._log_predictions(batch.melspec_gen, batch.transcript)
            self._log_scalars(self.valid_metrics)
            self._log_audio(batch.waveform_gen.squeeze(1), batch.transcript)

        for name, p in self.generator.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_audio(self, audio, text, examples_to_log=3):
        self.writer.add_audio(
            'audio', audio[:examples_to_log], caption=text[:examples_to_log], sample_rate=self.sr
        )

    def _log_predictions(self, spectrogram_batch, transcript_batch):
        if self.writer is None:
            return

        idx = random.randint(0, spectrogram_batch.size(0) - 1)
        spectrogram = spectrogram_batch[idx]
        text = transcript_batch[idx]
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.detach().cpu()))
        self.writer.add_image(
            "spectrogram", ToTensor()(image), caption=text
        )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.generator.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
