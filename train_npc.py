import torch.optim as optim
import time
import pickle
import os

from library.network import *
from library.util.dataloader import load_dataset
from library.util.utils import plot_

class NPC:
    def __init__(self, args, grad_scaler):
        self.args = args
        self.grad_scaler = grad_scaler
        self.time = time.time()

        # (x,y_tilde)
        Ttloader, self.len_train, self.len_test = load_dataset(self.args.data_name, batch_size=args.batch_size, dir=args.data_dir).train_test()
        self.trainloader, self.testloader = Ttloader['train'], Ttloader['test']

        # y_hat
        if args.dataset in ['MNIST','FMNIST','CIFAR10']:
            y_hat_path = os.path.join(self.args.cls_dir, self.args.data_name)
        else:
            y_hat_path = os.path.join(self.args.cls_dir, self.args.data_name+'.pk')
        with open(y_hat_path, 'rb') as f:
            y_hat_dict = pickle.load(f)
        self.y_hat = torch.tensor(y_hat_dict['Train']['label'])

        # y_prior
        with open(os.path.join(self.args.cls_dir, self.args.knn_mode + '_' + self.args.data_name), 'rb') as f:
            y_prior_dict = pickle.load(f)

        if 'onehot' in self.args.knn_mode:
            self.y_prior = torch.tensor(y_prior_dict['class'])
        else: # proba
            self.y_prior = torch.tensor(y_prior_dict['proba'])

        print('\n===> AE Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.AE = CVAE(self.args)
        self.AE.to(self.device)
        self.optimizer = optim.Adam(self.AE.parameters(), lr=self.args.lr)

    def generate_alpha_from_original_proba(self, y_prior):
        if self.args.knn_mode=='onehot':
            proba = torch.zeros(len(y_prior), self.args.n_classes).to(self.device)
            return proba.scatter(1,y_prior.view(-1, 1), self.args.prior_norm)+1.0+1/self.args.n_classes
        elif self.args.knn_mode=='proba': # proba
            if self.args.selected_class == self.args.n_classes:
                return self.args.prior_norm * y_prior + 1.0 + 1 / self.args.n_classes
            else: # topk
                values, indices = torch.topk(y_prior, k=int(self.args.selected_class), dim=1)
                y_prior_temp = Variable(torch.zeros_like(y_prior)).to(self.device)
                return self.args.prior_norm * y_prior_temp.scatter(1, indices, values) + 1.0 + 1 / self.args.n_classes

    def loss_function(self, y_tilde_data, y_tilde_recon, alpha_prior, alpha_infer):
        recon_loss = nn.BCEWithLogitsLoss(reduction='sum')(y_tilde_recon, y_tilde_data)

        KL = torch.sum(torch.lgamma(alpha_prior) - torch.lgamma(alpha_infer) + (alpha_infer - alpha_prior) *
                         torch.digamma(alpha_infer), dim=1)

        return recon_loss, torch.sum(KL)

    def update_model(self):
        self.AE.train()
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
        batch = 0

        for indexes, images, _, _ in self.trainloader:
            images = images.to(self.device)
            label_one_hot = torch.zeros(len(indexes), self.args.n_classes)
            label_one_hot = label_one_hot.scatter(1, self.y_hat[indexes].view(-1, 1), 1).to(self.device)
            alpha_prior = self.generate_alpha_from_original_proba(self.y_prior[indexes].to(self.device))

            with torch.cuda.amp.autocast():
                y_recon, alpha_infer = self.AE(images, label_one_hot)
                recon_loss, kl_loss = self.loss_function(label_one_hot, y_recon, alpha_prior, alpha_infer)
                loss = recon_loss + self.args.beta * kl_loss

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

            batch+=1
            if batch%100==0:
                print(batch, loss.item())

        time_elapse = time.time() - self.time

        return epoch_loss, epoch_recon, epoch_kl, time_elapse

    def save_result(self, train_loss_total, train_loss_recon, train_loss_kl):
        self.train_loss.append(train_loss_total/self.len_train)
        self.train_recon_loss.append(train_loss_recon/self.len_train)
        self.train_kl_loss.append(train_loss_kl/self.len_train)

        print('Train', train_loss_total/self.len_train, train_loss_recon/self.len_train)

        return

    def key_name(self, k):
        if self.args.model == 'CNN_MNIST':
            last_layer = 'linear3'
        elif self.args.model == 'CNN_CIFAR':
            last_layer = 'l_c1'
        elif self.args.model == 'Resnet50Pre':
            last_layer = 'model'
        else:
            last_layer = None

        res = k.split('.')

        if self.args.class_method == 'causalNL':
            return res[0] == 'y_encoder' and (res[1] != last_layer if 'CNN' in self.args.model else res[1] == last_layer)
        else:
            return res[0] != last_layer if 'CNN' in self.args.model else res[0] == last_layer

    def run(self):
        # initialize
        self.train_loss, self.train_recon_loss, self.train_kl_loss = [], [], []

        # update feature extractor
        old_dict = torch.load(self.args.model_dir + 'classifier.pk')
        change_dict = self.AE.state_dict()

        if self.args.class_method == 'causalNL':
            k_base_num = 10
        else:
            k_base_num = 0
        k_add_num = 6

        if 'CNN' in self.args.model:
            old_dict = {'FE.' + k[k_base_num:]: v for k, v in old_dict.items() if self.key_name(k)}
        elif 'Resnet50' in self.args.model:
            old_dict = {'FE.' + k[k_base_num + k_add_num:]: v for k, v in old_dict.items() if self.key_name(k)}

        change_dict.update(old_dict)
        self.AE.load_state_dict(change_dict)

        # train model using train dataset
        for epoch in range(self.args.total_iter):
            train_loss, train_recon, train_kl, time_train = self.update_model()
            # print result loss
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train)
            self.save_result(train_loss, train_recon, train_kl)

        plot_(self.args.gen_dir, self.train_loss, 'train_loss')
        plot_(self.args.gen_dir, self.train_recon_loss, 'train_recon_loss')
        plot_(self.args.gen_dir, self.train_kl_loss, 'train_kl_loss')

        torch.save(self.AE.state_dict(), self.args.gen_model_dir+'_AE.pk')

        return

