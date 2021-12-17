import torch
from torch.nn import MSELoss


class MSELossGenerator(MSELoss):
    def forward(self, doutput, _=None):
        """
        GAN generator G loss
        :param doutput: multi discriminator MD outputs
        :param _: to keep signature the same
        :return: loss
        """
        loss = 0
        for x in doutput:
            loss += super().forward(x, torch.ones(x.size(), device=x.device))
        return loss
