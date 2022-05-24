#############################################################################
# Original paper: Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels (NIPS 18) https://proceedings.neurips.cc/paper/2018/file/a19744e268754fb0148b017647355b7b-Paper.pdf
# Official code: https://github.com/bhanML/Co-teaching
#############################################################################
from network import *

import torch.optim as optim
import numpy as np
import time
import os

from util.dataloader import load_dataset
from util.utils import plot_, save_data

class Coteaching:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.data_name = self.args.data_name
        self.time = time.time()

        self.dataset = load_dataset(self.data_name, batch_size=args.batch_size, dir=args.data_dir)
        Ttloader, self.len_train, self.len_test = self.dataset.train_test()
        self.trainloader, self.testloader = Ttloader['train'], Ttloader['test']

        print('\n===> Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.args.model == 'CNN_MNIST':
            self.net1 = CNN_MNIST(self.n_classes, self.args.dropout)
            self.net2 = CNN_MNIST(self.n_classes, self.args.dropout)
        elif self.args.model == 'CNN_CIFAR':
            self.net1 = CNN(self.n_classes, self.args.dropout)
            self.net2 = CNN(self.n_classes, self.args.dropout)
        elif self.args.model == 'Resnet50Pre':
            self.net1 = ResNet50Pre(self.n_classes, self.args.dropout)
            self.net2 = ResNet50Pre(self.n_classes, self.args.dropout)

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
        elif self.args.dataset in ['CIFAR100', 'Animal']:
            lr_decay_start = 100
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

    def loss_function(self, forget_rate, output1, output2, labels):
        loss_1 = self.metric(output1, labels)
        loss_2 = self.metric(output2, labels)

        _, net1_index = torch.sort(loss_1)
        _, net2_index = torch.sort(loss_2)
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(labels))
        net1_index, net2_index = net1_index[:num_remember], net2_index[:num_remember]
        loss1 = self.criterion(output1[net2_index], labels[net2_index])
        loss2 = self.criterion(output2[net1_index], labels[net1_index])

        return loss1, loss2

    def update_model(self, epoch):
        self.net1.train()
        self.net2.train()
        self.adjust_learning_rate(self.optimizer1, epoch)
        self.adjust_learning_rate(self.optimizer2, epoch)
        epoch_loss, epoch_class_accuracy1, epoch_class_accuracy2, epoch_label_accuracy1, epoch_label_accuracy2 = 0, 0, 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, outputs1 = self.net1(images)
            _, outputs2 = self.net2(images)
            # loss
            loss1, loss2 = self.loss_function(self.rate_schedule[epoch], outputs1, outputs2, labels)

            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()
            epoch_loss += loss1.item() * len(labels)

            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()
            epoch_loss += loss2.item() * len(labels)

            # accuracy
            _, model_label1 = torch.max(outputs1, dim=1)
            _, model_label2 = torch.max(outputs2, dim=1)
            epoch_class_accuracy1 += (classes == model_label1.cpu()).sum().item()
            epoch_label_accuracy1 += (labels == model_label1).cpu().sum().item()
            epoch_class_accuracy2 += (classes == model_label2.cpu()).sum().item()
            epoch_label_accuracy2 += (labels == model_label2).cpu().sum().item()

        epoch_class_accuracy = max(epoch_class_accuracy1,epoch_class_accuracy2)
        epoch_label_accuracy = max(epoch_label_accuracy1,epoch_label_accuracy2)
        time_elapse = time.time() - self.time
        return epoch_loss/2, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def evaluate_model(self):
        # calculate test accuracy
        self.net1.eval()
        self.net2.eval()
        epoch_class_accuracy1, epoch_class_accuracy2 = 0, 0
        for index, images, classes, _ in self.testloader:
            images = images.to(self.device)
            classes = classes.to(self.device)
            _, outputs1 = self.net1(images)
            _, outputs2 = self.net2(images)
            # accuracy
            model_label1 = np.argmax(outputs1.detach().cpu().numpy(), axis=1)
            model_label2 = np.argmax(outputs2.detach().cpu().numpy(), axis=1)
            epoch_class_accuracy1 += (classes.cpu().numpy() == model_label1).sum().item()
            epoch_class_accuracy2 += (classes.cpu().numpy() == model_label2).sum().item()

        epoch_class_accuracy = max(epoch_class_accuracy1, epoch_class_accuracy2)
        time_elapse = time.time() - self.time
        return epoch_class_accuracy, int(epoch_class_accuracy1<epoch_class_accuracy2) ,time_elapse

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

        plot_(self.args.cls_dir, self.loss_train, 'train_loss')
        plot_(self.args.cls_dir, self.train_class_acc, 'train_class_accuracy')
        plot_(self.args.cls_dir, self.train_label_acc, 'train_label_accuracy')
        plot_(self.args.cls_dir, self.test_acc, 'test_accuracy')

        # save one network with higher test accuracy
        if network_name==0:
            net = self.net1
        else:
            net= self.net2

        save_data(os.path.join(self.args.cls_dir, self.data_name), self.dataset, self.device, net)
        torch.save(net.state_dict(), self.args.model_dir + 'classifier.pk')

        return 0, self.train_class_acc[-1], self.train_label_acc[-1], self.test_acc[-1], epoch
