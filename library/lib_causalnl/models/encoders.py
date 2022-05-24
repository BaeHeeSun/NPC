
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


__all__ = ["CONV_Decoder_FMNIST","CONV_Encoder_FMNIST", "CONV_T_Decoder","Y_Encoder","Z_Encoder", "X_Decoder", "T_Decoder","CONV_Encoder_CIFAR","CONV_Decoder_CIFAR", "CONV_Encoder_CLOTH1M","CONV_Decoder_CLOTH1M","CONV_T_Decoder_CLOTH1M"]



def make_hidden_layers(num_hidden_layers=1, hidden_size=5, prefix="y"):
    block = nn.Sequential()
    for i in range(num_hidden_layers):
        block.add_module(prefix+"_"+str(i), nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.BatchNorm1d(hidden_size),nn.ReLU()))
    return block



class CONV_Encoder_FMNIST(nn.Module):
    def __init__(self, in_channels =1, feature_dim = 28, num_classes = 2,  hidden_dims = [32, 64, 128, 256], z_dim = 2):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*4, z_dim)

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),1,self.feature_dim ,self.feature_dim )
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


class CONV_Decoder_FMNIST(nn.Module):

    def __init__(self, num_classes = 2, hidden_dims = [256, 128, 64, 32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim + num_classes, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               ),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 4),
                            nn.Sigmoid())


    def forward(self, z, y_hat):
        out = torch.cat((z, y_hat), dim=1)
        out = self.decoder_input(out)
        out = out.view(-1, 256, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out



class CONV_Encoder_CIFAR(nn.Module):
    def __init__(self, in_channels =3, feature_dim = 32, num_classes = 2,  hidden_dims = [32, 64, 128, 256], z_dim = 2):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*4, z_dim)

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),self.in_channels,self.feature_dim,self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


class CONV_Decoder_CIFAR(nn.Module):

    def __init__(self, num_classes = 2, hidden_dims = [256, 128, 64,32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim + num_classes, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)


        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1),
                            nn.Sigmoid())


    def forward(self, z, y_hat):
        out = torch.cat((z, y_hat), dim=1)
        out = self.decoder_input(out)
        out = out.view(-1, 256, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out


class CONV_T_Decoder(nn.Module):
    def __init__(self, in_channels =1, feature_dim = 28, num_classes = 10,  hidden_dims = [32, 64, 128, 256]):
        super().__init__()

        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*4, num_classes)
   

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),self.in_channels,self.feature_dim,self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        n_logits = self.fc(x)
        return n_logits


class CONV_T_Decoder_CLOTH1M(nn.Module):
    def __init__(self, in_channels =3, feature_dim = 224, num_classes = 14,  hidden_dims = [32, 64, 128, 256, 512]):
        super().__init__()

        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 7, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*16, num_classes)
   

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),self.in_channels,self.feature_dim,self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        n_logits = self.fc(x)
        return n_logits




class CONV_Encoder_CLOTH1M(nn.Module):
    def __init__(self, in_channels =3, feature_dim = 224, num_classes = 2,  hidden_dims = [32, 64, 128, 256, 512], z_dim = 2):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 7, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*16, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*16, z_dim)

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),self.in_channels,self.feature_dim,self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = self.encoder(x)
     
        x = torch.flatten(x, start_dim=1)
       
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


class CONV_Decoder_CLOTH1M(nn.Module):

    def __init__(self, num_classes = 2, hidden_dims = [512, 256, 128, 64, 32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim + num_classes, hidden_dims[0] * 16)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=7,
                                       stride = 2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)


        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=9,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1),
                            nn.Sigmoid())


    def forward(self, z, y_hat):
        out = torch.cat((z, y_hat), dim=1)
        out = self.decoder_input(out)   
        out = out.view(-1, 512, 4, 4)
        out = self.decoder(out)  
        out = self.final_layer(out)
        return out





class T_Decoder(nn.Module):
    def __init__(self, feature_dim = 2, num_classes = 2, num_hidden_layers=1, hidden_size = 5):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.T_fc1 = nn.Linear(feature_dim+num_classes, hidden_size)
        self.T_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="T")
        self.T_fc2 = nn.Linear(hidden_size, num_classes)   

    def forward(self, x, y_hat):
        x = x.view(-1, self.feature_dim)
        out = torch.cat((x, y_hat), dim=1)
        out = F.relu(self.T_fc1(out))
        out = self.T_h_layers(out)
        n_logits = self.T_fc2(out)
        return n_logits


class Y_Encoder(nn.Module):
    def __init__(self, feature_dim = 2, num_classes = 2, num_hidden_layers=1, hidden_size = 5):
        super().__init__()
        self.y_fc1 = nn.Linear(feature_dim, hidden_size)
        self.y_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="y")
        self.y_fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = F.relu(self.y_fc1(x)) 
        out = self.y_h_layers(out)
        c_logits = self.y_fc2(out)
        return c_logits


class Z_Encoder(nn.Module):
    def __init__(self, feature_dim = 2, num_classes = 2, num_hidden_layers=1, hidden_size = 5, z_dim = 2):
        super().__init__()
        self.z_fc1 = nn.Linear(feature_dim+num_classes, hidden_size)
        self.z_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="z")
        self.z_fc_mu = nn.Linear(hidden_size, z_dim)  # fc21 for mean of Z
        self.z_fc_logvar = nn.Linear(hidden_size, z_dim)  # fc22 for log variance of Z

    def forward(self, x, y_hat):
        out = torch.cat((x, y_hat), dim=1)
        out = F.relu(self.z_fc1(out))
        out = self.z_h_layers(out)
        mu = F.elu(self.z_fc_mu(out))
        logvar = F.elu(self.z_fc_logvar(out))
        return mu, logvar


class X_Decoder(nn.Module):
    def __init__(self, feature_dim = 2, num_classes = 2, num_hidden_layers=1, hidden_size = 5, z_dim = 1):
        super().__init__()
        self.recon_fc1 = nn.Linear(z_dim+num_classes, hidden_size)
        self.recon_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="recon")
        self.recon_fc2 = nn.Linear(hidden_size, feature_dim)    

    def forward(self, z, y_hat):
        out = torch.cat((z, y_hat), dim=1)
        out = F.relu(self.recon_fc1(out))
        out = self.recon_h_layers(out)
        x = self.recon_fc2(out)
        return x




