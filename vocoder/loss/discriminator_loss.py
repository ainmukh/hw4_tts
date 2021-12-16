import torch
from torch.nn import MSELoss


class MSELossDiscriminator(MSELoss):
    def forward(self, doutput_gen, doutput_real):
        """
        GAN discriminator D loss
        :param doutput_gen: discriminator outputs of multi discriminator
        :param doutput_real:
        :return: loss
        """
        loss = 0
        for gen, real in zip(doutput_gen, doutput_real):
            loss_gen = super().forward(gen, torch.zeros(gen.size()))
            loss_real = super().forward(real, torch.ones(real.size()))
            loss += loss_gen + loss_real
        return loss
