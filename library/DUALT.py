#############################################################################
# Original paper: Dual T: Reducing Estimation Error for Transition Matrix in Label-noise Learning (NIPS 20) https://proceedings.neurips.cc/paper/2020/file/512c5cad6c37edb98ae91c8a76c3a291-Paper.pdf
# Official code: https://github.com/a5507203/Dual-T
#############################################################################
from network import *

import torch.optim as optim
import numpy as np
import time
import os

from util.dataloader import load_dataset
from util.utils import plot_, save_data

class DualT:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.data_name = self.args.data_name
        self.time = time.time()

        self.dataset = load_dataset(self.data_name, batch_size=args.batch_size, dir=args.data_dir)
        vloader, self.len_train, self.len_val, self.len_test = self.dataset.train_val_test()
        self.trainloader,self.validloader,self.testloader = vloader['train'],vloader['valid'],vloader['test']

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

    def update_pre_model(self):
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
        return acc/self.len_val

    def est_t_matrix(self,eta_corr,filter_outlier):
        T = np.empty((self.n_classes, self.n_classes))

        # find a 'perfect example' for each class
        for i in np.arange(self.n_classes):

            if not filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)

            T[i] = eta_corr[idx_best]

        return torch.tensor(T).float()

    def update_transition_matrix(self):
        self.transition_spade = torch.zeros(self.n_classes, self.n_classes)
        p = []
        for idx, images, _, labels in self.trainloader:
            images = images.to(self.device)
            _, outputs = self.net(images)
            outputs = F.softmax(outputs, dim=1)
            _, index = torch.max(outputs, dim=1)
            index = index.cpu()
            for i in range(len(idx)):
                self.transition_spade[int(index[i])][int(labels[i])]+=1
            p += outputs.cpu().data.numpy()[:].tolist()

        self.transition_spade/=torch.sum(self.transition_spade,dim=1).view(-1,1)+1e-6
        p = np.array(p)
        self.transition_club = self.est_t_matrix(p, filter_outlier = True)

        self.transition = torch.matmul(self.transition_spade,self.transition_club).to(self.device)
        return

    def loss_function(self, outputs, labels):
        loss = 0.
        out_softmax = F.softmax(outputs, dim=1)
        for i in range(len(labels)):
            temp_softmax = torch.unsqueeze(out_softmax[i], 0)
            pro1 = out_softmax[i][labels[i]]
            out_T = torch.matmul(self.transition.t(), temp_softmax.t())
            pro2 = out_T[labels[i]]
            beta = Variable(pro1 / pro2, requires_grad=True)
            temp = torch.unsqueeze(outputs[i], 0)
            cross_loss = F.cross_entropy(temp, torch.unsqueeze(labels[i], 0))
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(labels)

    def update_model(self):
        self.net.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
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
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

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
        valid_acc = 0
        # train model to find anchor point
        for epoch in range(self.args.total_epochs):
            valid_new = self.early_stop()
            if valid_new < valid_acc:
                epoch_test_acc, _ = self.evaluate_model()
                break

            valid_acc = valid_new
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_pre_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, epoch_class_acc/self.len_train)
        pre_acc = epoch_test_acc

        # generate transition matrix
        self.update_transition_matrix()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

        # train model
        for epoch in range(self.args.total_epochs):
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
        
        return pre_acc, self.train_class_acc[-1], self.train_label_acc[-1], self.test_acc[-1], epoch

