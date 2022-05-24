#############################################################################
# Original paper: Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization (CVPR 20) https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.html
# Official code: https://github.com/hongxin001/JoCoR
#############################################################################

from network import *

import torch.optim as optim
import numpy as np
import time
import os

from util.dataloader import load_dataset
from util.utils import plot_, save_data

class JoCor:
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
        self.optimizer = optim.Adam(list(self.net1.parameters())+list(self.net2.parameters()), lr=self.args.lr)

    def define_configuration(self):
        self.metric = nn.CrossEntropyLoss(reduction='none').to(self.device)
        forget_rate = self.args.noisy_ratio
        if self.args.noise_type == 'asym':
            forget_rate/=2
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

    def kl_loss_compute(self, pred, soft_targets, reduce=True):
        kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False)

        if reduce:
            return torch.mean(torch.sum(kl, dim=1))
        else:
            return torch.sum(kl, 1)

    def loss_function(self, forget_rate, output1, output2, labels):
        loss_1 = self.metric(output1, labels)
        loss_2 = self.metric(output2, labels)
        kl_1 = self.kl_loss_compute(output1, output2,False)
        kl_2 = self.kl_loss_compute(output2, output1, False)

        loss = (loss_1+loss_2)+(kl_1+kl_2)
        _, index = torch.sort(loss)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(labels))
        net_index = index[:num_remember]
        loss_up = torch.mean(loss[net_index])

        return loss_up

    def update_model(self, epoch):
        self.net1.train()
        self.net2.train()
        self.adjust_learning_rate(self.optimizer, epoch)
        epoch_loss, epoch_class_accuracy1, epoch_class_accuracy2, epoch_label_accuracy1, epoch_label_accuracy2 = 0, 0, 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, outputs1 = self.net1(images)
            _, outputs2 = self.net2(images)
            # loss
            loss = self.loss_function(self.rate_schedule[epoch], outputs1, outputs2, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)

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
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

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

        if network_name==0:
            net = self.net1
        else:
            net= self.net2

        save_data(os.path.join(self.args.cls_dir, self.data_name), self.dataset, self.device, net)
        torch.save(net.state_dict(), self.args.model_dir + 'classifier.pk')

        return 0, self.train_class_acc[-1], self.train_label_acc[-1], self.test_acc[-1], epoch
