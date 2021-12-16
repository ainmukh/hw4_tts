from .generator_loss import MSELossGenerator as GeneratorLoss
from .discriminator_loss import MSELossDiscriminator as DiscriminatorLoss
from .feature_loss import L1LossFeature as FeatureLoss

__all__ = ['GeneratorLoss', 'DiscriminatorLoss', 'FeatureLoss']
