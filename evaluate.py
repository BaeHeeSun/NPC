import time
import pickle

from library.network import *
import library.lib_causalnl.models as models
from library.util.dataloader import load_dataset

# From classifier to autoencoder
# Get P(y^|x) and P(x|y^,y)
# Make P(y|y^) and P(y^) ==> Make P(y|x)

class Acc_calculator:
    def __init__(self, args):
        self.args = args
        self.n_classes = args.n_classes
        self.time = time.time()

        # dataloader
        Ttloader, _, self.len_test = load_dataset(self.args.data_name, batch_size=args.batch_size, dir=args.data_dir).train_test()
        self.testloader = Ttloader['test']

        print('\n===> Acc Calculator Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def merge_classifier_and_autoencoder(self):
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

        self.net.load_state_dict(torch.load(self.args.model_dir + 'classifier.pk'))
        self.net.to(self.device)

        # Load Generator
        self.AE = CVAE(self.args)
        self.AE.load_state_dict(torch.load(self.args.gen_model_dir+'_AE.pk'))
        self.AE.to(self.device)

        # final accuracy evaluation
        self.net.eval()
        self.AE.eval()

        accuracy = 0
        for _, images, classes, _ in self.testloader:
            batch_size = images.shape[0]
            images = images.to(self.device)
            # classifier side ==> P(y^|x)
            if self.args.class_method == 'causalNL':
                _, _, _, _, _, output, _ = self.net(images)
            else:
                _, output = self.net(images)

            p_y_tilde = F.softmax(output, dim=1).detach().cpu()

            # autoencoder side ==> P(y|y^,x)
            p_y_bar_x_y_tilde = torch.zeros(batch_size, self.n_classes, self.n_classes)
            for lab in range(self.n_classes):
                # generate noisy label
                label_one_hot = torch.zeros(batch_size, self.n_classes)
                label_one_hot[:, lab] = 1
                label_one_hot = label_one_hot.to(self.device)
                _, alpha_infer = self.AE(images, label_one_hot)
                alpha_infer = alpha_infer.detach().cpu() - 1.0
                p_y_bar_x_y_tilde[:, :, lab] = alpha_infer / torch.sum(alpha_infer, dim=1).view(-1, 1)

                del label_one_hot

            # P(y|y^,x)*P(y^|x)=P(y,y^|x)
            p_y_expansion = p_y_tilde.reshape(batch_size, 1, self.n_classes).repeat([1, self.n_classes, 1])  # batch*class*label
            p_y_y_tilde = p_y_bar_x_y_tilde * p_y_expansion  # batch*class*label
            _, pseudo_label = torch.max(torch.sum(p_y_y_tilde, dim=2), dim=1)
            accuracy += torch.sum(classes == pseudo_label).item()

        return accuracy / self.len_test