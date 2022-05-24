import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from .encoders import *
from network import *

__all__ = ["VAE_CNN_MNIST", "VAE_CNN_CIFAR", "VAE_Resnet50Pre" ]

class BaseVAE(nn.Module):
    def __init__(self, feature_dim=28, num_hidden_layers=1, hidden_size=25, z_dim =10, num_classes=10  ):
        super().__init__()
        self.y_encoder = Y_Encoder(feature_dim =feature_dim, num_classes = num_classes, num_hidden_layers=num_hidden_layers+10, hidden_size = hidden_size)
        self.z_encoder = Z_Encoder(feature_dim=feature_dim, num_classes=num_classes, num_hidden_layers=num_hidden_layers, hidden_size = hidden_size, z_dim=z_dim)
        self.x_decoder = X_Decoder(feature_dim=feature_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, hidden_size = hidden_size, z_dim=z_dim)
        self.t_decoder = T_Decoder(feature_dim=feature_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, hidden_size = hidden_size)
        self.kl_divergence = None
        self.flow  = None
    def _y_hat_reparameterize(self, c_logits):
        return F.gumbel_softmax(c_logits)

    def _z_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x): 
        ### trick 1, add a softmax function to logits
        c_embedding, c_logits = self.y_encoder(x)
        y_hat = self._y_hat_reparameterize(c_logits)
        mu, logvar = self.z_encoder(x, y_hat)
        z = self._z_reparameterize(mu, logvar)
        # self.kl_divergence = self._kld(z, (mu, logvar))
        x_hat = self.x_decoder(z,y_hat)
        n_logits = self.t_decoder(x_hat, y_hat)

        return x_hat, n_logits, mu, logvar, c_embedding, c_logits, y_hat
        # return x_hat, n_logits, mu, logvar, c_logits, y_hat

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z

        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)
        

        kl = qz - pz

        return kl

class VAE_CNN_MNIST(BaseVAE):
    def __init__(self, feature_dim=28, input_channel=1, z_dim=10, num_classes=10):
        super().__init__()

        self.y_encoder = CNN_MNIST(num_classes, dropout=0.)
        self.z_encoder = CONV_Encoder_FMNIST(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_FMNIST(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder(feature_dim=feature_dim, in_channels=input_channel, num_classes=num_classes)


class VAE_CNN_CIFAR(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim=25, num_classes=10):
        super().__init__()

        self.y_encoder = CNN(num_classes, dropout_rate=0.)
        self.z_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder(feature_dim=feature_dim, in_channels=input_channel, num_classes=num_classes)

class VAE_Resnet50Pre(BaseVAE):
    def __init__(self, feature_dim=224, input_channel=3, z_dim=100, num_classes=14):
        super().__init__()

        self.y_encoder = ResNet50Pre(num_classes, dropout=0.)
        self.z_encoder = CONV_Encoder_CLOTH1M(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CLOTH1M(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder_CLOTH1M(feature_dim=feature_dim, in_channels=input_channel, num_classes=num_classes)

