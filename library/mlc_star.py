from network import *

import torch.optim as optim
import numpy as np
import pickle
import time
import os

from util.dataloader import load_dataset
from util.utils import plot_, save_data

class DummyScheduler(optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])
        return lrs

    def step(self, epoch=None):
        pass

class MLC_star:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.data_name = self.args.data_name
        self.time = time.time()

        self.dataset = load_dataset(self.data_name, batch_size=args.batch_size, dir=args.data_dir)
        mTvtloader, self.len_meta, self.len_train, self.len_val, self.len_test = self.dataset.meta_train_val_test()
        self.metaloader, self.trainloader, self.validloader, self.testloader = mTvtloader['meta'], mTvtloader['train'], mTvtloader['valid'], mTvtloader['test']

        print('\n===> Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # y_hat
        # Load Classifier
        if self.args.class_method == 'causalNL':
            self.classifier = models.__dict__["VAE_" + self.args.model](z_dim=self.args.causalnl_z_dim, num_classes=self.args.n_classes)
        else:
            if self.args.model == 'CNN_MNIST':
                self.classifier = CNN_MNIST(self.n_classes, self.args.dropout)
            elif self.args.model == 'CNN_CIFAR':
                self.classifier = CNN(self.n_classes, self.args.dropout)
            elif self.args.model == 'Resnet50Pre':
                self.classifier = ResNet50Pre(self.n_classes, self.args.dropout)

        self.classifier.load_state_dict(torch.load(self.args.model_dir + 'classifier.pk'))
        self.classifier.to(self.device)

        if self.args.model == 'CNN_MNIST':
            self.net = CNN_MNIST(self.n_classes, self.args.dropout)
            self.metanet = MetaNet(256, 64, 128, self.n_classes)
            self.args.mlt_epoch = 120
        elif self.args.model == 'CNN_CIFAR':
            self.net = CNN(self.n_classes, self.args.dropout)
            self.metanet = MetaNet(128, 64, 128, self.n_classes)
            self.args.mlt_epoch = 120
        elif self.args.model == 'Resnet50Pre':
            self.net = ResNet50Pre(self.n_classes, self.args.dropout)
            self.metanet = MetaNet(2048, 64, 128, self.n_classes)
            if self.args.dataset == 'Food':
                self.args.mlt_epoch = 60
            else: # clothing
                self.args.mlt_epoch = 10

        self.net.to(self.device)
        self.metanet.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # mean

        self.main_optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.meta_optimizer = optim.Adam(self.metanet.parameters(), lr=3e-4, weight_decay=0, amsgrad=True)

        self.main_schdlr = optim.lr_scheduler.MultiStepLR(self.main_optimizer, milestones=[80,100], gamma=0.1)
        self.meta_schdlr = DummyScheduler(self.meta_optimizer)

    def _concat(self, xs):
        return torch.cat([x.view(-1) for x in xs])

    def soft_cross_entropy(self, logit, pseudo_target, reduction='mean'):
        loss = -(pseudo_target * torch.nn.functional.log_softmax(logit, -1)).sum(-1)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError('Invalid reduction: %s' % reduction)

    def update_params(self, params, grads, eta, opt, deltaonly=False, return_s=False):
       with torch.no_grad():
           return self.update_params_sgd(params, grads, eta, opt, deltaonly, return_s)

    def update_params_sgd(self, params, grads, eta, opt, deltaonly, return_s=False):
        # supports SGD-like optimizers
        ans = []

        if return_s:
            ss = []
        wdecay = opt.defaults['weight_decay']

        for i, param in enumerate(params):
            dparam = grads[i] + param * wdecay  # s=1
            s = 1

            if deltaonly:
                ans.append(- dparam * eta)
            else:
                ans.append(param - dparam * eta)

            if return_s:
                ss.append(s * eta)

        if return_s:
            return ans, ss
        else:
            return ans

    def update_model(self):
        self.net.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, _ in self.trainloader:
            index_meta, image_meta, class_meta, _ = next(self.metaloader)

            image_meta, class_meta = image_meta.to(self.device), class_meta.to(self.device)
            images = images.to(self.device)
            _, logit_classifier = self.classifier(images)
            _, labels = torch.max(logit_classifier, dim=1)
            
            eta =self.main_schdlr.get_lr()[-1]

            # meta
            _, logit_meta = self.net(image_meta)
            loss_meta = self.criterion(logit_meta, class_meta)
            gw = torch.autograd.grad(loss_meta, self.net.parameters())
            
            # label correction
            embeddings, outputs = self.net(images)
            pseudo = self.metanet(embeddings.detach(), labels)
            loss = self.soft_cross_entropy(outputs,pseudo)

            f_param_grads = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
            f_params_new, dparam_s = self.update_params(self.net.parameters(), f_param_grads, eta, self.main_optimizer, return_s=True)

            # set w as w'
            f_param = []
            for i, param in enumerate(self.net.parameters()):
                f_param.append(param.data.clone())
                param.data = f_params_new[i].data
            # Hessian approximation
            Hw = 1

            # compute d_w' L_{D}(w')
            _, logit_meta = self.net(image_meta)
            loss_meta = self.criterion(logit_meta, class_meta)
            gw_prime = torch.autograd.grad(loss_meta, self.net.parameters())

            tmp1 = [(1 - Hw * dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
            gw_norm2 = (self._concat(gw).norm()) ** 2
            tmp2 = [gw[i] / gw_norm2 for i in range(len(gw))]
            gamma = torch.dot(self._concat(tmp1), self._concat(tmp2))
            Lgw_prime = [dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]

            proxy_g = -torch.dot(self._concat(f_param_grads), self._concat(Lgw_prime))

            # back prop on alpha
            self.meta_optimizer.zero_grad()
            proxy_g.backward()

            for i, param in enumerate(self.metanet.parameters()):
                if param.grad is not None:
                    param.grad.add_(gamma*self.dw_prev[i])
                    self.dw_prev[i] = param.grad.clone()

            self.meta_optimizer.step()
            self.dw_prev = [0 for param in self.metanet.parameters()]

            for i, param in enumerate(self.net.parameters()):
                param.data = f_param[i]
                param.grad = f_param_grads[i].data
            self.main_optimizer.step()

            epoch_loss+=loss_meta+loss
            # accuracy
            _, outputs = self.net(images)
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

        self.main_schdlr.step()
        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def evaluate_model(self):
        # calculate test accuracy
        self.net.eval()
        epoch_class_accuracy = 0
        for index, images, classes, _ in self.testloader:
            images = images.to(self.device)
            classes = classes.to(self.device)
            _, outputs = self.net(images)
            # accuracy
            model_label = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            epoch_class_accuracy += (classes.cpu().numpy() == model_label).sum().item()

        time_elapse = time.time() - self.time
        return epoch_class_accuracy, time_elapse

    def save_result(self, epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc):
        self.loss_train.append(epoch_loss / self.len_train)
        self.train_class_acc.append(epoch_class_acc / self.len_train)
        self.train_label_acc.append(epoch_label_acc / self.len_train)
        self.test_acc.append(epoch_test_acc / self.len_test)

        print('Train', epoch_class_acc / self.len_train, epoch_label_acc / self.len_train)
        print('Test', epoch_test_acc / self.len_test)

        return

    def run(self):
        # initialize
        self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc = [], [], [], []
        self.dw_prev = [0 for param in self.metanet.parameters()]

        # train model
        for epoch in range(self.args.mlt_epoch):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.save_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        plot_(self.args.gen_dir, self.loss_train, 'train_loss')
        plot_(self.args.gen_dir, self.train_class_acc, 'train_class_accuracy')
        plot_(self.args.gen_dir, self.train_label_acc, 'train_label_accuracy')
        plot_(self.args.gen_dir, self.test_acc, 'test_accuracy')

        torch.save(self.net.state_dict(), self.args.gen_model_dir + 'classifier.pk')
        torch.save(self.metanet.state_dict(), self.args.gen_model_dir + 'metanet.pk')

        return 0, self.train_class_acc[-1], self.train_label_acc[-1], self.test_acc[-1], epoch
