import torch
from torch.nn import L1Loss


class L1LossFeature(L1Loss):
    def forward(self, feature_gen, feature_real):
        """
        Feature Matching loss
        :param feature_gen: |multi discriminator| x |discriminator layers| features of generated wav
        :param feature_real: |multi discriminator| x |discriminator layers| features of real wav
        :return: loss
        """
        loss = 0
        for d_gen, d_real in zip(feature_gen, feature_real):
            for f_gen, f_real in zip(d_gen, d_real):
                loss += super().forward(f_gen, f_real)
        return loss
