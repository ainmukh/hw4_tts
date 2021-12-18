import torch
import torch.nn as nn
from torch.nn import MSELoss


class DiscriminatorLoss_(MSELoss):
    def forward(self, doutput_gen, doutput_real):
        """
        GAN discriminator D loss
        :param doutput_gen: discriminator outputs of multi discriminator
        :param doutput_real:
        :return: loss
        """
        loss = 0
        for gen, real in zip(doutput_gen, doutput_real):
            loss_gen = (gen**2).mean()
            loss_real = ((real - 1)**2).mean()
            # loss_gen = super().forward(gen, torch.zeros(gen.size(), device=gen.device))
            # loss_real = super().forward(real, torch.ones(real.size(), device=real.device))
            loss += loss_gen + loss_real
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, doutput_gen, doutput_real):
        loss = 0
        for gen, real in zip(doutput_gen, doutput_real):
            loss_gen = (gen ** 2).mean()
            loss_real = ((real - 1) ** 2).mean()
            loss += loss_gen + loss_real
        return loss
