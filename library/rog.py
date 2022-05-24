#############################################################################
# Original paper: Robust Inference via Generative Classifiers for Handling Noisy Labels (ICML 19) http://proceedings.mlr.press/v97/lee19f/lee19f.pdf
# Official code: https://github.com/pokaxpoka/RoGNoisyLabel
#############################################################################

import torch.optim as optim
import numpy as np
import time
import os
import sklearn.covariance
import scipy

from network import *
import lib_causalnl.models as models
from util.dataloader import load_dataset

class ROG:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.time = time.time()

        self.dataset = load_dataset(self.args.data_name, batch_size=args.batch_size, dir=args.data_dir)
        vloader, self.len_train, self.len_val, self.len_test = self.dataset.train_val_test()
        self.trainloader, self.testloader = vloader['train'], vloader['test']
        self.validloader = vloader['valid']

        print('\n===> Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        self.criterion = nn.CrossEntropyLoss().to(self.device) # mean
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

    def random_sample_mean(self, feature, total_label, num_classes):
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        new_feature, fraction_list = [], []
        frac = 0.7
        sample_mean_per_class = torch.Tensor(num_classes, feature.size(1)).fill_(0).cuda()
        total_label = total_label.cuda()

        total_selected_list = []
        for i in range(num_classes):
            index_list = total_label.eq(i)
            temp_feature = feature[index_list.nonzero(), :]
            temp_feature = temp_feature.view(temp_feature.size(0), -1)
            shuffler_idx = torch.randperm(temp_feature.size(0))
            index = shuffler_idx[:int(temp_feature.size(0) * frac)]
            fraction_list.append(int(temp_feature.size(0) * frac))
            total_selected_list.append(index_list.nonzero()[index.cuda()])
            selected_feature = torch.index_select(temp_feature, 0, index.cuda())
            new_feature.append(selected_feature)
            sample_mean_per_class[i].copy_(torch.mean(selected_feature, 0))

        total_covariance = 0
        for i in range(num_classes):
            flag = 0
            X = 0
            for j in range(fraction_list[i]):
                temp_feature = new_feature[i][j]
                temp_feature = temp_feature - sample_mean_per_class[i]
                temp_feature = temp_feature.view(-1, 1)
                if flag == 0:
                    X = temp_feature.transpose(0, 1)
                    flag = 1
                else:
                    X = torch.cat((X, temp_feature.transpose(0, 1)), 0)
                # find inverse
            group_lasso.fit(X.cpu().numpy())
            inv_sample_conv = group_lasso.covariance_
            inv_sample_conv = torch.from_numpy(inv_sample_conv).float().cuda()
            if i == 0:
                total_covariance = inv_sample_conv * fraction_list[i]
            else:
                total_covariance += inv_sample_conv * fraction_list[i]
            total_covariance = total_covariance / sum(fraction_list)
        new_precision = scipy.linalg.pinvh(total_covariance.cpu().numpy())
        new_precision = torch.from_numpy(new_precision).float().cuda()

        return sample_mean_per_class, new_precision, total_selected_list

    def MCD_single(self, feature, sample_mean, inverse_covariance):
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        temp_batch = 100
        total, mahalanobis_score = 0, 0
        frac = 0.7
        for data_index in range(int(np.ceil(feature.size(0) / temp_batch))):
            temp_feature = feature[total: total + temp_batch].cuda()
            gaussian_score = 0
            batch_sample_mean = sample_mean
            zero_f = temp_feature - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, inverse_covariance), zero_f.t()).diag()
            # concat data
            if total == 0:
                mahalanobis_score = term_gau.view(-1, 1)
            else:
                mahalanobis_score = torch.cat((mahalanobis_score, term_gau.view(-1, 1)), 0)
            total += temp_batch

        mahalanobis_score = mahalanobis_score.view(-1)
        feature = feature.view(feature.size(0), -1)
        _, selected_idx = torch.topk(mahalanobis_score, int(feature.size(0) * frac))
        selected_feature = torch.index_select(feature, 0, selected_idx.cuda())
        new_sample_mean = torch.mean(selected_feature, 0)

        # compute covariance matrix
        X = 0
        flag = 0
        for j in range(selected_feature.size()[0]):
            temp_feature = selected_feature[j]
            temp_feature = temp_feature - new_sample_mean
            temp_feature = temp_feature.view(-1, 1)
            if flag == 0:
                X = temp_feature.transpose(0, 1)
                flag = 1
            else:
                X = torch.cat((X, temp_feature.transpose(0, 1)), 0)
        # find inverse
        group_lasso.fit(X.cpu().numpy())
        new_sample_cov = group_lasso.covariance_

        return new_sample_mean, new_sample_cov, selected_idx

    def make_validation(self, feature, total_label, sample_mean, inverse_covariance, num_classes):
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        temp_batch = 100
        total, mahalanobis_score, prediction = 0, 0, 0
        frac = 0.5
        feature = feature.cuda()
        for data_index in range(int(np.floor(feature.size(0) / temp_batch))):
            temp_feature = feature[total: total + temp_batch]
            temp_label = total_label[total: total + temp_batch]
            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[i]
                zero_f = temp_feature - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, inverse_covariance), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1, 1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
            generative_out = torch.index_select(gaussian_score, 1, temp_label.cuda()).diag()
            # concat data
            if total == 0:
                mahalanobis_score = generative_out
            else:
                mahalanobis_score = torch.cat((mahalanobis_score, generative_out), 0)
            total += temp_batch

        _, selected_idx = torch.topk(mahalanobis_score, int(total * frac))
        return selected_idx

    def test_softmax(self, model, total_data, total_label):
        model.eval()

        batch_size = self.args.batch_size

        total, correct_D = 0, 0

        for data_index in range(int(np.floor(total_data.size(0) / batch_size))):
            data, target = total_data[total: total + batch_size], total_label[total: total + batch_size]
            data, target = Variable(data, volatile=True), Variable(target)

            if self.args.class_method == 'causalNL':
                _, _, _, _, _, total_out, _ = model(data.to(self.device))
            else:
                _, total_out = model(data.to(self.device))

            total += batch_size
            pred = torch.max(total_out, dim=1)[1].cpu().detach()
            equal_flag = pred.eq(target.data).cpu()
            correct_D += equal_flag.sum()

        return 100. * correct_D / total

    def train_weights(self, G_soft_list, total_val_data, total_val_label):

        batch_size = self.args.batch_size
        # loss function
        nllloss = nn.NLLLoss().cuda()

        # parameyer
        num_ensemble = len(G_soft_list)
        train_weights = torch.Tensor(num_ensemble, 1).fill_(1).cuda()
        train_weights = nn.Parameter(train_weights)
        total, correct_D = 0, 0
        optimizer = optim.Adam([train_weights], lr=0.02)
        total_epoch = 10
        total_num_data = total_val_data[0].size(0)

        for data_index in range(int(np.floor(total_num_data / batch_size))):
            target = total_val_label[total: total + batch_size].cuda()
            target = Variable(target)
            soft_weight = F.softmax(train_weights, dim=0)
            total_out = 0

            for i in range(num_ensemble):
                out_features = total_val_data[i][total: total + batch_size].cuda()
                out_features = Variable(out_features, volatile=True)
                feature_dim = out_features.size(1)
                output = F.softmax(G_soft_list[i](out_features), dim=1)

                output = Variable(output.data, volatile=True)
                if i == 0:
                    total_out = soft_weight[i] * output
                else:
                    total_out += soft_weight[i] * output

            total += batch_size
            pred = total_out.data.max(1)[1]
            equal_flag = pred.eq(target.data).cpu()
            correct_D += equal_flag.sum()

        for epoch in range(total_epoch):
            total = 0
            shuffler_idx = torch.randperm(total_num_data)

            print('This is ',epoch, '-th epoch!')

            for data_index in range(int(np.floor(total_num_data / batch_size))):
                index = shuffler_idx[total: total + batch_size]
                target = torch.index_select(total_val_label, 0, index).cuda()
                target = Variable(target)
                total += batch_size

                def closure():
                    optimizer.zero_grad()
                    soft_weight = F.softmax(train_weights, dim=0)

                    total_out = 0
                    for i in range(num_ensemble):
                        out_features = torch.index_select(total_val_data[i], 0, index).cuda()
                        out_features = Variable(out_features)
                        feature_dim = out_features.size(1)
                        output = F.softmax(G_soft_list[i](out_features), dim=1)

                        if i == 0:
                            total_out = soft_weight[i] * output
                        else:
                            total_out += soft_weight[i] * output

                    total_out = torch.log(total_out + 10 ** (-10))
                    loss = nllloss(total_out, target)
                    loss.backward()
                    return loss

                optimizer.step(closure)

        correct_D, total = 0, 0

        for data_index in range(int(np.floor(total_num_data / batch_size))):
            target = total_val_label[total: total + batch_size].cuda()
            target = Variable(target)
            soft_weight = F.softmax(train_weights, dim=0)
            total_out = 0

            for i in range(num_ensemble):
                out_features = total_val_data[i][total: total + batch_size].cuda()
                out_features = Variable(out_features, volatile=True)
                feature_dim = out_features.size(1)
                output = F.softmax(G_soft_list[i](out_features), dim=1)

                output = Variable(output.data, volatile=True)
                if i == 0:
                    total_out = soft_weight[i] * output
                else:
                    total_out += soft_weight[i] * output

            total += batch_size
            pred = total_out.data.max(1)[1]
            equal_flag = pred.eq(target.data).cpu()
            correct_D += equal_flag.sum()

        soft_weight = F.softmax(train_weights, dim=0)
        return soft_weight

    def run(self):
        train_data_list = [torch.zeros(self.len_train,self.emb_dim), torch.zeros(self.len_train,self.n_classes)]
        test_val_data_list = [torch.zeros(self.len_val,self.emb_dim), torch.zeros(self.len_val,self.n_classes)]
        train_label_list,test_val_label_list = torch.zeros(self.len_train).long(), torch.zeros(self.len_val).long()

        if self.args.model == 'CNN_MNIST':
            total_test_data, total_test_label = torch.zeros(self.len_test,1, 28,28), torch.zeros(self.len_test).long()
        elif self.args.model == 'CNN_CIFAR':
            total_test_data, total_test_label = torch.zeros(self.len_test, 3, 32, 32), torch.zeros(self.len_test).long()
        else:
            total_test_data, total_test_label = torch.zeros(self.len_test, 3, 224, 224), torch.zeros(self.len_test).long()

        for index, images, _, labels in self.trainloader:
            images = images.to(self.device)
            if self.args.class_method == 'causalNL':
                _, _, _, _, feature, output, _ = self.net(images)
            else:
                feature, output = self.net(images)

            train_data_list[0][index] = feature.cpu().detach()
            train_data_list[1][index] = output.cpu().detach()
            train_label_list[index] = labels

        for index, images, _, labels in self.validloader:
            images = images.to(self.device)
            if self.args.class_method == 'causalNL':
                _, _, _, _, feature, output, _ = self.net(images)
            else:
                feature, output = self.net(images)

            test_val_data_list[0][index] = feature.cpu().detach()
            test_val_data_list[1][index] = output.cpu().detach()
            test_val_label_list[index] = labels

        for index, images, classes, _ in self.testloader:
            total_test_data[index] = images.cpu().detach()
            total_test_label[index] = classes

        layer_list = list(range(2))
        print('Random Sample Mean')
        sample_mean_list, sample_precision_list = [], []
        for index in range(len(layer_list)):
            sample_mean, sample_precision, _ = self.random_sample_mean(train_data_list[index].cuda(), train_label_list.cuda(), self.n_classes)
            sample_mean_list.append(sample_mean)
            sample_precision_list.append(sample_precision)

        print('Single MCD and merge the parameters')
        new_sample_mean_list = []
        new_sample_precision_list = []
        for index in range(len(layer_list)):
            new_sample_mean = torch.Tensor(self.n_classes, train_data_list[index].size()[1]).fill_(0).cuda()
            new_covariance = 0
            for i in range(self.n_classes):
                index_list = train_label_list.eq(i)
                temp_feature = train_data_list[index][index_list.nonzero(), :]
                temp_feature = temp_feature.view(temp_feature.size(0), -1)
                temp_mean, temp_cov, _ \
                    = self.MCD_single(temp_feature.cuda(), sample_mean_list[index][i], sample_precision_list[index])
                new_sample_mean[i].copy_(temp_mean)
                if i == 0:
                    new_covariance = temp_feature.size(0) * temp_cov
                else:
                    new_covariance += temp_feature.size(0) * temp_cov

            new_covariance = new_covariance / train_data_list[index].size()[0]
            new_precision = scipy.linalg.pinvh(new_covariance)
            new_precision = torch.from_numpy(new_precision).float().cuda()
            new_sample_mean_list.append(new_sample_mean)
            new_sample_precision_list.append(new_precision)

        G_soft_list = []
        target_mean = new_sample_mean_list
        target_precision = new_sample_precision_list
        for i in range(len(new_sample_mean_list)):
            dim_feature = new_sample_mean_list[i].size(1)
            sample_w = torch.mm(target_mean[i], target_precision[i])
            sample_b = -0.5 * torch.mm(torch.mm(target_mean[i], target_precision[i]), target_mean[i].t()).diag() + torch.Tensor(self.n_classes).fill_(
                np.log(1. / self.n_classes)).cuda()
            G_soft_layer = nn.Linear(int(dim_feature), self.n_classes).cuda()
            G_soft_layer.weight.data.copy_(sample_w)
            G_soft_layer.bias.data.copy_(sample_b)
            G_soft_list.append(G_soft_layer)

        print('Construct validation set')
        sel_index = -1
        selected_list = self.make_validation(test_val_data_list[sel_index], test_val_label_list,
                                              target_mean[sel_index], target_precision[sel_index], self.n_classes)
        new_val_data_list = []
        for i in range(len(new_sample_mean_list)):
            new_val_data = torch.index_select(test_val_data_list[i], 0, selected_list.cpu())
            new_val_label = torch.index_select(test_val_label_list, 0, selected_list.cpu())
            new_val_data_list.append(new_val_data)

        soft_weight = self.train_weights(G_soft_list, new_val_data_list, new_val_label)
        soft_acc = self.test_softmax(self.net, total_test_data, total_test_label)

        print('softmax accuracy: ' + str(soft_acc))

        return soft_acc.item()
