from .gan_loss import MSELossGenerator as GANLoss
from .feature_loss import L1LossFeature as FMLoss
from .melspec_loss import L1LossMelSpec as MelSpecLoss

__all__ = ['GANLoss', 'FMLoss', 'MelSpecLoss']
