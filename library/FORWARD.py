#############################################################################
# Original paper:Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach (CVPR 17) https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf
# Official code: https://github.com/giorgiop/loss-correction
#############################################################################
from network import *

import torch.optim as optim
import numpy as np
import time
import os

from util.dataloader import load_dataset
from util.utils import plot_, save_data

class Forward:
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
        self.nll = nn.NLLLoss().to(self.device)
        self.optimizer = optim.Adagrad(self.net.parameters(), lr=self.args.lr)

        self.proba = torch.zeros(self.len_train, self.n_classes)

    def update_model(self):
        self.net.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, outputs = self.net(images)
            # loss
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

            self.proba[index] = F.softmax(outputs, dim=1).detach().cpu()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def update_transition_matrix(self):
        self.transition = torch.zeros(self.n_classes, self.n_classes)
        data_dict = {}
        for i in range(self.n_classes):
            data_dict[i] = []

        for idx, _, _,labels in self.trainloader:
            labels = labels
            for i in range(len(idx)):
                data_dict[labels[i].item()].append(idx[i])

        for i in range(self.n_classes):
            object_proba = self.proba[data_dict[i]][:,i]
            proba, index = torch.topk(object_proba, int(len(object_proba)*0.03))
            self.transition[i] = self.proba[data_dict[i]][index[-1]]
        
        self.transition = torch.transpose(self.transition,1,0)
        self.transition = self.transition.to(self.device)
        return

    def loss_forward(self, outputs, labels):
        outputs = F.softmax(outputs, dim=1)
        new_proba = torch.log(torch.clamp(self.transition[labels]*outputs,1e-6,1-1e-6))

        return self.nll(new_proba, labels)

    def update_model_forward(self):
        self.net.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, outputs = self.net(images)
            # loss
            loss = self.loss_forward(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

            self.proba[index] = F.softmax(outputs, dim=1).detach().cpu()

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
        self.test_acc.append(epoch_test_acc/self.len_test)

        print('Train', epoch_loss / self.len_train, epoch_class_acc / self.len_train, epoch_label_acc / self.len_train)
        print('Test', epoch_test_acc / self.len_test)

        return

    def run(self):
        # initialize
        self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc = [], [], [], []

        # train model
        self.update_model()
        for epoch in range(self.args.total_epochs):
            self.update_transition_matrix()
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_forward()
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
