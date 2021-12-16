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


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            vocoder,
            criterion,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            sr=22050
    ):
        super().__init__(generator, vocoder, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader

        self.melspec = MelSpectrogram(MelSpectrogramConfig()).to(device)
        self.melspec_silence = -11.5129251

        self.aligner = GraphemeAligner().to(device)

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
            "melspec_loss", 'length_loss', "grad norm", writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "melspec_loss", 'length_loss', 'loss', writer=self.writer
        )
        self.sr = sr

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        # TODO
        for tensor_for_gpu in [
            "tokens",
            "text_encoded"
        ]:
            batch.__dict__[tensor_for_gpu] = batch.__dict__[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_iteration(self, batch, epoch: int, batch_num: int):
        # batch = self.move_batch_to_device(batch, self.device)
        batch = batch.to(self.device)

        melspec = self.melspec(batch.waveform)
        durations = self.aligner(
            batch.waveform, batch.waveform_length, batch.transcript
        ).to(self.device)
        durations = (durations * batch.melspec_length.reshape(-1, 1))
        batch.melspec = melspec
        batch.durations = durations

        self.optimizer.zero_grad()
        batch = self.generator(batch)
        # if batch.durations.size() != batch.durations_pred.size():
        #     print(batch.durations.size(), batch.durations_pred.size(), batch.tokens.size())
        #     print(batch.transcript)
        #     return

        melspec_loss, length_loss = self.criterion(batch)  # TODO

        loss = melspec_loss + length_loss

        loss.backward()
        # self._clip_grad_norm()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.set_step((epoch - 1) * self.len_epoch + batch_num)
        self.train_metrics.update('melspec_loss', melspec_loss.item())
        self.train_metrics.update('length_loss', length_loss.item())
        self.train_metrics.update("grad norm", self.get_grad_norm())

        if batch_num % self.log_step == 0 and batch_num:
            self.writer.add_scalar(
                "learning rate", self.lr_scheduler.get_last_lr()[0]
            )
            self._log_predictions(batch.melspec_pred, batch.transcript)
            # self._log_attention(batch.attn)
            self._log_scalars(self.train_metrics)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
                # tqdm(self.data_loader, desc="train", total=len(self.data_loader))
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
        self.vocoder.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader), desc="validation",
                    total=len(self.valid_data_loader)
            ):
                # for batch_idx, batch in enumerate(self.valid_data_loader):
                # batch = self.move_batch_to_device(batch, self.device)
                batch = batch.to(self.device)

                melspec = self.melspec(batch.waveform)
                durations = self.aligner(
                    batch.waveform, batch.waveform_length, batch.transcript
                ).to(self.device)
                durations = (durations * batch.melspec_length.reshape(-1, 1))
                batch.melspec = melspec
                batch.durations = durations

                batch = self.generator(batch)

                audio = []
                for i in range(batch.melspec_pred.size(0)):
                    wav = self.vocoder.inference(batch.melspec_pred[i].unsqueeze(0))
                    audio.append(wav)
                batch.audio = torch.cat(audio, 0)

                melspec_loss, length_loss = self.criterion(batch)
                loss = melspec_loss + length_loss

                self.valid_metrics.update('melspec_loss', melspec_loss.item(), n=batch.melspec_pred.size(0))
                self.valid_metrics.update('length_loss', length_loss.item(), n=batch.melspec_pred.size(0))
                self.valid_metrics.update('loss', loss.item(), n=batch.melspec_pred.size(0))

            self.writer.set_step(epoch * self.len_epoch, "valid")
            self._log_predictions(batch.melspec_pred, batch.transcript)
            # self._log_attention(batch.attn)
            self._log_scalars(self.valid_metrics)
            self._log_audio(batch.audio, batch.transcript)

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

    def _log_attention(self, attn_batch):
        attention = random.choice(attn_batch)
        image = PIL.Image.open(plot_spectrogram_to_buf(attention.detach().cpu().log()))
        self.writer.add_image("attention", ToTensor()(image), caption='')

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

    def test(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            for batch_idx, batch in enumerate(
                    tqdm(self.data_loader, desc="train", total=self.len_epoch)
                    # tqdm(self.data_loader, desc="train", total=len(self.data_loader))
            ):
                batch = batch.to(self.device)

                durations = self.aligner(
                    batch.waveform, batch.waveform_length, batch.transcript
                ).to(self.device)
                batch.durations = durations
                if batch.tokens.size(-1) != batch.durations.size(-1):
                    print('found a problem')
                    with open(f'saved_{batch_idx}.json', 'w') as fout:
                        fout.write(json.dumps([s for s in batch.transcript]))
