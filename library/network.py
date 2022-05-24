import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

############################################################################################################
# Convolutional NN
class CNN_MNIST(nn.Module):
    def __init__(self, n_classes, dropout=0.0):
        super(CNN_MNIST, self).__init__()
        self.fc1 = nn.Conv2d(1,8,3,padding=1)
        self.fc2 = nn.Conv2d(8,16,3,padding=1)
        self.bn = nn.BatchNorm2d(8)

        self.linear1 = nn.Linear(16*28*28, 28*28)
        self.linear2 = nn.Linear(28*28, 256)
        self.linear3 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.bn(F.relu(self.fc1(x)))
        x = self.tanh(self.fc2(x))
        x = x.view(x.size(0), -1)
        output1 = self.dropout(self.linear1(x))
        output2 = self.linear2(output1)
        output3 = self.linear3(output2)

        return output2, output3

############################################################################################################
class CNN(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.0):
        self.dropout_rate = dropout_rate
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(3,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_classes)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(self.bn1(h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(self.bn2(h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(self.bn3(h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(self.bn4(h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(self.bn5(h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(self.bn6(h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(self.bn7(h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(self.bn8(h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(self.bn9(h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        x=self.l_c1(h)

        return h, x

############################################################################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, dropout, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.dropout = nn.Dropout2d(p=dropout)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out

############################################################################################################
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

############################################################################################################
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dropout, in_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.dropout, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.maxpool(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = self.avgpool(out)
            out = torch.flatten(out,1)
            logit = self.fc(out)

        return out, logit

############################################################################################################
# 3. pretrain resnet
class ResNetPre(nn.Module):
    def __init__(self, block, num_blocks, dropout):
        super(ResNetPre, self).__init__()
        self.in_planes = 64
        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, 1000)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.dropout, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.maxpool(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = self.avgpool(out)
            out = torch.flatten(out,1)
            logit = self.fc(out)
        return out, logit

############################################################################################################
class classifier(nn.Module):
    def __init__(self,num_classes, dropout):
        super(classifier, self).__init__()

        self.model = ResNetPre(Bottleneck, [3,4,6,3], dropout)
        state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        self.model.load_state_dict(state_dict)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        feature, x = self.model(x)
        out = self.classifier(x)

        return feature, out

def ResNet50Pre(num_classes=10, dropout=0.0): # resnet 50 as pretrain true
    # clothing1m/food101n
    return classifier(num_classes, dropout)

############################################################################################################
class CVAE(nn.Module):
    def __init__(self, args):
        super(CVAE, self).__init__()
        self.args = args
        if self.args.model == 'CNN_MNIST':
            self.FE = CNN_MNIST(args.n_classes, args.dropout)
            self.encoder = nn.Linear(args.n_classes + args.n_classes, args.n_classes)
            self.decoder = nn.Linear(args.n_classes + args.n_classes, args.n_classes)
        elif self.args.model == 'CNN_CIFAR':
            self.FE = CNN(args.n_classes, args.dropout)
            self.encoder = nn.Linear(args.n_classes + args.n_classes, args.n_classes)
            self.decoder = nn.Linear(args.n_classes + args.n_classes, args.n_classes)
        elif self.args.model == 'Resnet50Pre':
            self.FE = ResNetPre(Bottleneck, [3, 4, 6, 3], args.dropout)
            self.encoder = nn.Linear(1000 + args.n_classes, args.n_classes)
            self.decoder = nn.Linear(1000 + args.n_classes, args.n_classes)

        self.softplus = nn.Softplus(self.args.softplus_beta)

    def Encoder(self,x,y):
        _, x = self.FE(x)
        input = torch.cat([x, y], 1)
        alpha = self.softplus(self.encoder(input)) + 1.0 + 1 / self.args.n_classes # to make probability

        return alpha

    def reparameterize(self, alpha):
        return ((torch.ones_like(alpha) / alpha) * (torch.log(torch.rand_like(alpha) * alpha) + torch.lgamma(alpha))).exp()

    def Decoder(self, x, y):
        _, x = self.FE(x)
        y_tilde = self.decoder(torch.cat([x,y],1))

        return y_tilde

    def forward(self,x,ytilde):
        alpha = self.Encoder(x,ytilde)
        z = self.reparameterize(alpha)
        y_tilde = self.Decoder(x,z)

        return y_tilde, alpha

############################################################################################################
class MetaNet(nn.Module): # for metanet
    def __init__(self, hx_dim, cls_dim, h_dim, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.hdim = h_dim
        self.cls_emb = nn.Embedding(self.num_classes, cls_dim)

        in_dim = hx_dim + cls_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.num_classes + int(False), bias=(not False))
        )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.cls_emb.weight)
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.xavier_normal_(self.net[4].weight)

        self.net[0].bias.data.zero_()
        self.net[2].bias.data.zero_()
        self.net[4].bias.data.zero_()

    def get_alpha(self):
        return torch.zeros(1)

    def forward(self, hx, y):
        y_emb = self.cls_emb(y)
        hin = torch.cat([hx, y_emb], dim=-1)

        logit = self.net(hin)
        out = F.softmax(logit, -1)

        return out

