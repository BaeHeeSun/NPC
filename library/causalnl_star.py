from network import *

import torch.optim as optim
import numpy as np
import pickle
import time
import os
import lib_causalnl.models as models

from util.dataloader import load_dataset
from util.utils import plot_, save_data

class CausalNL_star:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.time = time.time()

        self.dataset = load_dataset(self.args.data_name, batch_size=args.batch_size, dir=args.data_dir)
        Ttloader, self.len_train, self.len_test = self.dataset.train_test()
        self.trainloader, self.testloader = Ttloader['train'], Ttloader['test']

        # data given by classifier
        if args.dataset in ['MNIST', 'FMNIST', 'CIFAR10']:
            data_load_name = args.cls_dir + args.data_name
        else:
            data_load_name = args.cls_dir + args.data_name + '.pk'
        with open(data_load_name, 'rb') as f:
            self.labels = pickle.load(f)
        self.y_hat = torch.tensor(self.labels['Train']['label'])

        print('\n===> Post processing causal star Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.net1 = models.__dict__["VAE_"+self.args.model](z_dim=self.args.causalnl_z_dim, num_classes=self.args.n_classes)
        self.net2 = models.__dict__["VAE_"+self.args.model](z_dim=self.args.causalnl_z_dim, num_classes=self.args.n_classes)

        self.net1.to(self.device)
        self.net2.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)  # mean
        self.optimizer1 = optim.Adam(self.net1.parameters(), lr=self.args.lr)
        self.optimizer2 = optim.Adam(self.net2.parameters(), lr=self.args.lr)

    def define_configuration(self):
        self.metric = nn.CrossEntropyLoss(reduction='none').to(self.device)
        forget_rate = self.args.noisy_ratio
        num_gradual = 10
        self.rate_schedule = np.ones(self.args.total_epochs) * forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)

        if self.args.dataset in ['MNIST','FMNIST','CIFAR10']:
            lr_decay_start = 80
        else:
            lr_decay_start = int(self.args.total_epochs * 0.6)

        self.alpha_plan = [self.args.lr]*self.args.total_epochs
        self.beta1_plan = [0.9]*self.args.total_epochs

        for i in range(lr_decay_start, self.args.total_epochs):
            self.alpha_plan[i] = float(self.args.total_epochs - i) / (self.args.total_epochs - lr_decay_start) * self.args.lr
            self.beta1_plan[i] = 0.1

        return

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
        return

    def log_standard_categorical(self, p, reduction="mean"):
        """
        Calculates the cross entropy between a (one-hot) categorical vector
        and a standard (uniform) categorical distribution.
        :param p: one-hot categorical distribution
        :return: H(p, u)
        """
        # Uniform prior over y
        prior = F.softmax(torch.ones_like(p), dim=1)
        prior.requires_grad = False

        cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

        if reduction == "mean":
            cross_entropy = torch.mean(cross_entropy)
        else:
            cross_entropy = torch.sum(cross_entropy)

        return cross_entropy

    def vae_loss(self, x_hat, data, n_logits, targets, mu, log_var, c_logits, h_c_label):
        # x loss
        l1 = 0.1 * F.mse_loss(x_hat, data, reduction="mean")

        # \tilde{y]} loss
        l2 = 0.1 * F.cross_entropy(n_logits, targets, reduction="mean")
        #  uniform loss for x
        l3 = -0.00001 * self.log_standard_categorical(h_c_label, reduction="mean")
        #  Gaussian loss for z
        l4 = - 0.01 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return (l1 + l2 + l3 + l4)

    def loss_coteaching(self, y_1, y_2, t, forget_rate):
        loss_1 = F.cross_entropy(y_1, t, reduce=False)
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduce=False)
        ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember

    def update_model(self, epoch):
        self.net1.train()
        self.net2.train()
        self.adjust_learning_rate(self.optimizer1, epoch)
        self.adjust_learning_rate(self.optimizer2, epoch)
        epoch_loss, epoch_class_accuracy1, epoch_label_accuracy1 = 0, 0, 0
        for index, images, classes, _ in self.trainloader:
            images = images.to(self.device)
            labels = self.y_hat[index].to(self.device)

            x_hat1, n_logits1, mu1, logvar1, c_embedding1, c_logits1, y_hat1 = self.net1(images)
            x_hat2, n_logits2, mu2, logvar2, c_embedding2, c_logits2, y_hat2 = self.net2(images)
            # loss
            vae_loss_1 = self.vae_loss(x_hat1, images, n_logits1, labels, mu1, logvar1, c_logits1, y_hat1)
            vae_loss_2 = self.vae_loss(x_hat2, images, n_logits2, labels, mu2, logvar2, c_logits2, y_hat2)
            co_loss_1, co_loss_2 = self.loss_coteaching(c_logits1, c_logits2, labels, self.rate_schedule[epoch])

            loss1 = co_loss_1 + vae_loss_1
            loss2 = co_loss_2 + vae_loss_2

            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()
            epoch_loss += loss1.item()

            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()
            epoch_loss += loss2.item()

            # accuracy
            _, model_label1 = torch.max(c_logits1, dim=1)
            _, model_label2 = torch.max(c_logits2, dim=1)
            epoch_class_accuracy1 += (classes == model_label1.cpu()).sum().item()
            epoch_label_accuracy1 += (labels == model_label1).cpu().sum().item()

        time_elapse = time.time() - self.time

        return epoch_loss/2, epoch_class_accuracy1, epoch_label_accuracy1, time_elapse

    def evaluate_model(self):
        # calculate test accuracy
        self.net1.eval()
        self.net2.eval()
        epoch_class_accuracy1, epoch_class_accuracy2 = 0, 0
        with torch.no_grad():
            for index, images, classes, _ in self.testloader:
                images = images.to(self.device)
                classes = classes.to(self.device)

                x_hat1, _, _, _, _, c_logits1, y_hat1 = self.net1(images)
                x_hat2, _, _, _, _, c_logits2, y_hat2 = self.net2(images)
                # accuracy
                model_label1 = np.argmax(c_logits1.detach().cpu().numpy(), axis=1)
                model_label2 = np.argmax(c_logits2.detach().cpu().numpy(), axis=1)
                epoch_class_accuracy1 += (classes.cpu().numpy() == model_label1).sum().item()
                epoch_class_accuracy2 += (classes.cpu().numpy() == model_label2).sum().item()

        time_elapse = time.time() - self.time
        return epoch_class_accuracy1, int(epoch_class_accuracy1<epoch_class_accuracy2) ,time_elapse

    def save_result(self, epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc):
        self.loss_train.append(epoch_loss / self.len_train)
        self.train_class_acc.append(epoch_class_acc / self.len_train)
        self.train_label_acc.append(epoch_label_acc / self.len_train)
        self.test_acc.append(epoch_test_acc/self.len_test)

        print('Train', epoch_loss / self.len_train, epoch_class_acc / self.len_train, epoch_label_acc / self.len_train)
        print('Test', epoch_test_acc / self.len_test)

        return

    def run(self):
        # initialize
        self.define_configuration()

        self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc = [], [], [], []
        self.epoch_acc_clean, self.epoch_class_acc_noisy, self.epoch_label_acc_noisy = [], [], []

        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model(epoch)
            epoch_test_acc, network_name, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.save_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        plot_(self.args.gen_dir, self.loss_train, 'train_loss')
        plot_(self.args.gen_dir, self.train_class_acc, 'train_class_accuracy')
        plot_(self.args.gen_dir, self.train_label_acc, 'train_label_accuracy')
        plot_(self.args.gen_dir, self.test_acc, 'test_accuracy')

        torch.save(self.net1.state_dict(), self.args.gen_model_dir + 'classifier1.pk')
        torch.save(self.net2.state_dict(), self.args.gen_model_dir + 'classifier2.pk')

        return 0, self.train_class_acc[-1], self.train_label_acc[-1], self.test_acc[-1], epoch
