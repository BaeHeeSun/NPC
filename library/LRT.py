#############################################################################
# Original paper: Error-Bounded Correction of Noisy Labels (ICML 20) # Original paper: Error-Bounded Correction of Noisy Labels (ICML 20) http://proceedings.mlr.press/v119/zheng20c/zheng20c.pdf
# Official code: https://github.com/pingqingsheng/LRT
#############################################################################

from network import *

import torch.optim as optim
import numpy as np
import pickle
import time
import os
import math

from util.dataloader import load_dataset
from util.utils import plot_, save_data

class LRT:
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
            self.net = CNN_MNIST(self.n_classes, self.args.dropout)
        elif self.args.model == 'CNN_CIFAR':
            self.net = CNN(self.n_classes, self.args.dropout)
        elif self.args.model == 'Resnet50Pre':
            self.net = ResNet50Pre(self.n_classes, self.args.dropout)

        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # mean
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

    def learning_rate(self, init, optimizer, epoch):
        optim_factor = 0
        if (epoch > 200):
            optim_factor = 4
        elif (epoch > 160):
            optim_factor = 3
        elif (epoch > 120):
            optim_factor = 2
        elif (epoch > 60):
            optim_factor = 1

        for param_group in optimizer.param_groups:
            param_group['lr'] = init * math.pow(0.5, optim_factor)

        return

    def loss_function(self,outputs, softlabel):
        return -torch.sum(softlabel*torch.softmax(outputs,dim=1))

    def likelihood_ratio_test(self, outputs, labels):
        new_label = labels.clone()
        proba = torch.softmax(outputs, dim=1)
        label_proba = torch.gather(proba, 1, labels.view(-1,1)).view(-1)
        max_proba, pseudo = torch.max(proba, dim=1)
        change_idx = torch.where(max_proba/label_proba>99)[0]
        new_label[change_idx] = pseudo[change_idx]
        return new_label

    def update_softlabel(self, cur_epoch):
        self.net.eval()
        if cur_epoch == 0:
            self.soft_label = torch.zeros(self.len_train,self.n_classes).to(self.device)
        elif cur_epoch >= 24:
            eps = 1e-2
            for index, images, _, _  in self.trainloader:
                images = images.to(self.device)
                _, outputs = self.net(images)
                _, pseudo = torch.max(outputs, dim=1)
                softlabel = torch.ones(len(index), self.n_classes)*eps/(self.n_classes-1)
                softlabel = softlabel.to(self.device)
                softlabel.scatter_(1, pseudo.reshape(-1, 1), 1 - eps)
                self.soft_label[index] = softlabel

        return

    def update_model(self, cur_epoch):
        self.net.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, outputs = self.net(images)
            # loss
            if cur_epoch < 25:
                loss = self.criterion(outputs, labels)
            elif cur_epoch < 25+10:
                loss = self.loss_function(outputs, self.soft_label[index]) + self.criterion(outputs, labels)
            else:
                labels = self.likelihood_ratio_test(outputs, labels)
                loss = self.loss_function(outputs, self.soft_label[index]) + self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

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

        # train model
        for epoch in range(self.args.total_epochs):
            self.learning_rate(self.args.lr, self.optimizer, epoch)
            self.update_softlabel(epoch)
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model(epoch)
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.save_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        plot_(self.args.cls_dir, self.loss_train, 'train_loss')
        plot_(self.args.cls_dir, self.train_class_acc, 'train_class_accuracy')
        plot_(self.args.cls_dir, self.train_label_acc, 'train_label_accuracy')
        plot_(self.args.cls_dir, self.test_acc, 'test_accuracy')

        save_data(os.path.join(self.args.cls_dir, self.data_name), self.dataset, self.device, self.net)
        torch.save(self.net.state_dict(), self.args.model_dir + 'classifier.pk')

        return 0, self.train_class_acc[-1], self.train_label_acc[-1], self.test_acc[-1], epoch
