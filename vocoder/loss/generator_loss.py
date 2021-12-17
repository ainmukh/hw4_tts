import torch.nn as nn
from .generator_losses import GANLoss, FMLoss, MelSpecLoss


class GeneratorLoss(nn.Module):
    def __init__(self, lambda_fm: int, lambda_mel: int):
        super().__init__()
        self.gan_loss = GANLoss()
        self.fm_loss = FMLoss()
        self.mel_loss = MelSpecLoss()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def forward(self, batch):
        loss, gan_loss, fm_loss = 0, None, None
        if batch.mpd_gen and batch.msd_gen:
            mpd_output, msd_output = batch.mpd_gen, batch.msd_gen
            gan_loss = self.gan_loss(mpd_output) + self.gan_loss(msd_output)
            mpd_feat_gen, mpd_feat_real = batch.mpd_feat_gen, batch.mpd_feat_real
            msd_feat_gen, msd_feat_real = batch.msd_feat_gen, batch.msd_feat_real
            fm_loss = self.fm_loss(mpd_feat_gen, mpd_feat_real) + self.fm_loss(msd_feat_gen, msd_feat_real)

        mel_gen, mel_real = batch.melspec_gen, batch.melspec_real
        mel_loss = self.mel_loss(mel_gen, mel_real)

        loss = self.lambda_mel * mel_loss
        if gan_loss is not None:
            loss += gan_loss + self.lambda_fm * fm_loss
        return loss, fm_loss, mel_loss
