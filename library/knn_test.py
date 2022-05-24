from network import *
from util.dataloader import load_dataset
import lib_causalnl.models as models

import time
import os
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier

class KNN_tester:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.time = time.time()

        self.dataset = load_dataset(self.args.data_name, batch_size=args.batch_size, dir=args.data_dir)
        Tloader, self.len_train, self.len_test = self.dataset.train_test()
        self.trainloader, self.testloader = Tloader['train'], Tloader['test']
        
        # data given by classifier
        if args.dataset in ['MNIST','FMNIST','CIFAR10']:
            data_load_name = args.cls_dir+args.data_name
        else:
            data_load_name = args.cls_dir+args.data_name+'.pk'
        with open(data_load_name, 'rb') as f:
            self.labels = pickle.load(f)
        self.y_hat = self.labels['Train']['label']

        print('\n===> KNN model Start', torch.cuda.device_count())

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For each datapoint, get initial class name
    def knn_test(self):
        # Load Classifier
        if self.args.class_method == 'causalNL':
            self.net = models.__dict__["VAE_" + self.args.model](z_dim=self.args.causalnl_z_dim, num_classes=self.args.n_classes)
        else:
            if self.args.model == 'CNN_MNIST':
                self.net = CNN_MNIST(self.n_classes, self.args.dropout)
            elif self.args.model == 'CNN_CIFAR':
                self.net = CNN(self.n_classes, self.args.dropout)
            elif self.args.model == 'Resnet50Pre':
                self.net = ResNet50Pre(self.n_classes, self.args.dropout)

        if self.args.model == 'CNN_MNIST':
            self.emb_dim = 256
        elif self.args.model == 'CNN_CIFAR':
            self.emb_dim = 128
        elif self.args.model == 'Resnet50Pre':
            self.emb_dim = 2048

        self.net.load_state_dict(torch.load(self.args.model_dir +'classifier.pk'))
        self.net.to(self.device)

        # k-nn
        self.net.eval()
        # knn mode on
        neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
        embeddings, class_confi = [], []
        for idx, images, _, _ in self.trainloader:
            tmp_dict = {}
            images = images.to(self.device)
            labels = self.y_hat[idx]
            for i in range(self.n_classes):
                tmp_dict[i] = []
            for i, lbl in enumerate(labels):
                tmp_dict[lbl].append(i)
            if self.args.class_method == 'causalNL':
                _, _, _, _, mu, output, _ = self.net(images)
            else:
                mu, output = self.net(images)
            output = F.softmax(output, dim=1).cpu().detach()

            for i in range(self.n_classes):
                if len(tmp_dict[i]) == 0: 
                    continue
                tmp_array = torch.tensor(tmp_dict[i])
                _, index = torch.sort(torch.gather(output[tmp_array], 1, torch.tensor(labels)[tmp_array].view(-1, 1)).squeeze(1))
                embeddings.append(mu[tmp_array[index[-1]]].cpu().detach().tolist())
                class_confi.append(i)

        class_confi = np.array(class_confi)
        neigh.fit(embeddings, class_confi)
        print('Time : ', time.time() - self.time)

        # 2. predict class of test dataset
        acc = 0
        for index, images, classes, _ in self.testloader:
            images = images.to(self.device)
            if self.args.class_method == 'causalNL':
                _, _, _, _, mu, _, _ = self.net(images)
            else:
                mu, _ = self.net(images)
            model_output = neigh.predict(mu.cpu().detach().numpy())
            acc+=sum(classes.numpy()==model_output)

        print('Time : ', time.time() - self.time,'seed : ', self.args.seed, ',acc: ', acc, ', knn test finished')

        f = open(self.args.cls_dir + '_pre_method_' + str(self.args.seed) + "_Acc.txt", "w")
        f.write('======================================')
        f.write('\n')
        f.write("test acc : " + str(acc/self.len_test))
        f.write('\n')
        f.close()

        return

