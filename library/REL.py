#############################################################################
# Original paper: Robust early-learning: Hindering the memorization of noisy labels (ICLR 21) https://openreview.net/forum?id=Eql5b1_hTE4
# Official code: https://github.com/xiaoboxia/CDR
#############################################################################

from network import *

import torch.optim as optim
import numpy as np
import time
import os

from util.dataloader import load_dataset
from util.utils import plot_, save_data

from torch.optim.lr_scheduler import MultiStepLR

class REL:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.data_name = self.args.data_name
        self.time = time.time()

        self.dataset = load_dataset(self.data_name, batch_size=args.batch_size, dir=args.data_dir)
        vloader, self.len_train, self.len_val, self.len_test = self.dataset.train_val_test()
        self.trainloader, self.validloader, self.testloader = vloader['train'], vloader['valid'], vloader['test']

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
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=0.01, momentum=0.9)

    def adjust_learning_rate(self):
        self.clip = 1-self.args.noisy_ratio
        if self.args.model == 'CNN_MNIST':
            self.scheduler = MultiStepLR(self.optimizer, milestones=[10, 20], gamma=0.1)
        elif self.args.model in ['CNN_CIFAR', 'Resnet18']:
            self.scheduler = MultiStepLR(self.optimizer, milestones=[40, 80], gamma=0.1)

        return

    def update_model(self):
        self.net.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, outputs = self.net(images)
            # loss
            loss = self.criterion(outputs, labels)
            loss.backward()

            to_concat_g = []
            to_concat_v = []
            for name, param in self.net.named_parameters():
                if param.dim() in [2, 4]:
                    to_concat_g.append(param.grad.data.view(-1))
                    to_concat_v.append(param.data.view(-1))
            all_g = torch.cat(to_concat_g)
            all_v = torch.cat(to_concat_v)
            metric = torch.abs(all_g * all_v)
            num_params = all_v.size()[0]
            nz = int(self.clip * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1]

            for name, param in self.net.named_parameters():
                if param.dim() in [2, 4]:
                    mask = (torch.abs(param.data * param.grad.data) >= thresh).float().to(self.device)
                    mask = mask * self.clip
                    param.grad.data = mask * param.grad.data

            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs.detach(), dim=1)
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

    def early_stop(self):
        self.net.eval()
        acc = 0
        for _, images, _, labels in self.validloader:
            images = images.to(self.device)
            _, outputs = self.net(images)
            _, model_label = torch.max(outputs, dim=1)
            acc += (labels == model_label.cpu()).sum().item()
        return acc

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
        self.adjust_learning_rate()
        self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc = [], [], [], []
        valid_acc = 0

        # train model
        for epoch in range(self.args.total_epochs):
            valid_new = self.early_stop()
            if valid_new < valid_acc:
                break
            valid_acc = valid_new
            # self.scheduler.step()
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model()
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

