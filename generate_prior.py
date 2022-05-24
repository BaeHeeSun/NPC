import time
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

from library.network import *
import library.lib_causalnl.models as models
from library.util.dataloader import load_dataset

class KNN_Prior:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.time = time.time()

        self.dataset = load_dataset(self.args.data_name, batch_size=args.batch_size, dir=args.data_dir)
        Ttloader, self.len_train, self.len_test = self.dataset.train_test()
        self.trainloader, self.testloader = Ttloader['train'], Ttloader['test']

        # data given by classifier
        if args.dataset in ['MNIST','FMNIST','CIFAR10']:
            data_load_name = args.cls_dir+args.data_name
        else:
            data_load_name = args.cls_dir+args.data_name+'.pk'
        with open(data_load_name, 'rb') as f:
            self.labels = pickle.load(f)
        self.y_hat = self.labels['Train']['label']

        print('\n===> Prior Generation with KNN Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For each datapoint, get initial class name
    def get_prior(self):
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

        # 2. predict class of training dataset
        embeddings = np.zeros((self.len_train, self.emb_dim))
        for index, images, _, _ in self.trainloader:
            images = images.to(self.device)
            if self.args.class_method == 'causalNL':
                _, _, _, _, mu, _, _ = self.net(images)
            else:
                mu, _ = self.net(images)
            embeddings[index] = mu.cpu().detach().numpy()


        # onehot
        dict = {}
        model_output = neigh.predict(embeddings)
        dict['class'] = np.int64(model_output)
        with open(os.path.join(self.args.cls_dir, 'onehot_'+self.args.data_name), "wb") as f:
            pickle.dump(dict, f)
        f.close()
        print('Time : ', time.time() - self.time, 'class information saved')

        # proba
        dict = {}
        model_output = neigh.predict_proba(embeddings)
        if model_output.shape[1] < self.n_classes:
            tmp = np.zeros((model_output.shape[0], self.n_classes))
            tmp[:, neigh.classes_] = neigh.predict_proba(embeddings)
            dict['proba'] = tmp
        else:
            dict['proba'] = model_output  # data*n_class

        with open(os.path.join(self.args.cls_dir, 'proba_'+self.args.data_name), "wb") as f:
            pickle.dump(dict, f)
        f.close()

        print('Time : ', time.time() - self.time, 'proba information saved')


        return

