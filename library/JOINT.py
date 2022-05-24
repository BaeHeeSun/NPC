#############################################################################
# Original paper: Joint Optimization Framework for Learning with Noisy Labels (CVPR 18) https://openaccess.thecvf.com/content_cvpr_2018/papers/Tanaka_Joint_Optimization_Framework_CVPR_2018_paper.pdf
# Official code: https://github.com/DaikiTanaka-UT/JointOptimization
#############################################################################

from network import *

import torch.optim as optim
import numpy as np
import time
import os

from util.dataloader import load_dataset
from util.utils import plot_, save_data

class Joint:
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
        # self.criterion = nn.CrossEntropyLoss().to(self.device) # mean
        self.kl_loss = nn.KLDivLoss('batchmean').to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

    def generate_label(self):
        self.soft_label = torch.zeros(self.len_train, self.n_classes)
        for index, _, _, labels in self.trainloader:
            self.soft_label[index] = torch.zeros(len(index), self.n_classes).scatter(1, labels.view(-1, 1), 1.0)
        return

    def loss_function(self, output, labels):
        # output : logit
        # labels : soft label (float!)
        loss1 = self.kl_loss(torch.log_softmax(output, dim=1),labels)
        s_mean = torch.mean(F.softmax(output, dim=1), dim=0)
        loss2 = -torch.sum(torch.log(s_mean)/self.n_classes)
        loss3 =-torch.sum(F.softmax(output, dim=1)*torch.log_softmax(output, dim=1))/output.shape[0]

        return loss1+1.0*loss2+0.5*loss3

    def update_1st_phase(self):
        self.net.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, label in self.trainloader:
            images = images.to(self.device)
            labels = self.soft_label[index].to(self.device)
            _, outputs = self.net(images)
            # loss
            loss = self.loss_function(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (label == model_label.cpu()).sum().item()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def label_update(self):
        self.net.eval()
        for index,images, _, _ in self.trainloader:
            images = images.to(self.device)
            _, outputs = self.net(images)
            soft_label = F.softmax(outputs, dim=1)
            self.soft_label[index] = soft_label.detach().cpu()

        return

    def initialize_network(self):
        if self.args.model == 'CNN_MNIST':
            self.net = CNN_MNIST(self.n_classes, self.args.dropout)
        elif self.args.model == 'CNN_CIFAR':
            self.net = CNN(self.n_classes, self.args.dropout)
        elif self.args.model == 'Resnet50Pre':
            self.net = ResNet50Pre(self.n_classes, self.args.dropout)
        self.net.to(self.device)
        
        return
    
    def update_2nd_phase(self):
        self.net.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, label in self.trainloader:
            images = images.to(self.device)
            labels = self.soft_label[index].to(self.device)
            _, outputs = self.net(images)
            # loss
            loss = self.kl_loss(torch.log_softmax(outputs, dim=1),labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (label == model_label.cpu()).sum().item()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def evaluate_model(self):
        # calculate test accuracy
        self.net.eval()
        epoch_class_accuracy = 0
        for index, images, classes, _ in self.testloader:
            images = images.to(self.device)
            _, outputs = self.net(images)
            # accuracy
            _, model_label = torch.max(outputs.detach().cpu(), dim=1)
            epoch_class_accuracy += torch.sum(classes == model_label).item()

        time_elapse = time.time() - self.time
        return epoch_class_accuracy, time_elapse

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
        self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc = [], [], [], []

        # train model
        self.generate_label()
        #phase 1
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_1st_phase()
            if epoch>=69:
                self.label_update()

            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.save_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        # initialize network
        self.initialize_network()

        # phase 2
        for epoch in range(self.args.total_epochs, self.args.total_epochs+120):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_2nd_phase()
            epoch_test_acc, time_test = self.evaluate_model()
            if epoch in [self.args.total_epochs+39, self.args.total_epochs+79]:
                self.args.lr/=10
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
