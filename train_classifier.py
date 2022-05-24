import argparse
import os
import numpy as np
import torch
from library.util.utils import save_cls_text_configuration

# Make pretrained classifier and model prediction
from library.CE import CE
from library.JOINT import Joint
from library.COTEACHING import Coteaching
from library.JOCOR import JoCor
from library.ES import ES
from library.LS import LS
from library.SCE import SCE
from library.REL import REL
from library.FORWARD import Forward
from library.DUALT import DualT
from library.TVR import TVR
from library.CAUSALNL import CausalNL

# Label Correction Method
from library.LRT import LRT
from library.MLC import MLC

####################################################################################################################
parser = argparse.ArgumentParser(description="main")

# data condition
parser.add_argument('--dataset', type=str, default='MNIST', help = 'MNIST, FMNIST, CIFAR10, Food, Clothing')
parser.add_argument('--noise_type', type=str, default='clean', help='clean, sym, asym, idn, idnx')
parser.add_argument('--noisy_ratio', type=float, default=None, help='between 0 and 1')

# classifier condition
parser.add_argument('--class_method', type=str, default=None, help='classifier method')

# experiment condition
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--lr", type=float, default=0.001, help = "Learning rate (Default : 1e-3)")

# etc
parser.add_argument('--set_gpu', type=str, default='0', help='gpu setting')
parser.add_argument('--data_dir', type=str, default=None)
####################################################################################################################
if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.set_gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # define model
    if args.dataset in ['MNIST', 'FMNIST']:
        args.model = 'CNN_MNIST'
        args.dropout = 0.3
    elif args.dataset == 'CIFAR10':
        args.model = 'CNN_CIFAR'
        args.dropout = 0.3
    elif args.dataset in ['Clothing', 'Food']:
        args.model = 'Resnet50Pre'
        args.dropout = 0.0

    if args.class_method =='causalNL':
        args.dropout = 0.0

    # directory
    if args.noise_type == 'clean':
        args.data_name = args.dataset + '_00.0_' + args.noise_type + '.pk'
        data_noise = args.dataset + '_' + args.model + '_clean'
    else:
        args.data_name = args.dataset + '_' + str(100 * args.noisy_ratio) + '_' + args.noise_type + '.pk'
        data_noise = args.dataset + '_' + args.model + '_' + args.noise_type + '_' + str(100 * args.noisy_ratio)

    # data
    if args.dataset in ['MNIST', 'FMNIST', 'CIFAR10']:
        args.n_classes = 10
        args.causalnl_z_dim = 25
        args.batch_size = 128
        args.pre_epoch = 10
        if args.class_method in ['cores', 'rel']:
            args.total_epochs = 100
        elif args.class_method == 'causalNL':
            args.total_epochs = 150
        else:
            args.total_epochs = 200

    elif args.dataset in ['Clothing', 'Food']:
        if args.dataset == 'Clothing':
            args.n_classes = 14
            args.noisy_ratio = 0.38

            args.pre_epoch = 1
            args.total_epochs = 10

        elif args.dataset == 'Food': # food
            args.n_classes = 101
            args.noisy_ratio = 0.27

            args.pre_epoch = 5
            if args.class_method in ['cores', 'rel']:
                args.total_epochs = 50
            else:
                args.total_epochs = 100

        args.data_name = args.dataset
        args.noise_type = 'clean'
        args.causalnl_z_dim = 100
        args.batch_size = 64

    else:  # wrong data name
        print('Wrong dataset name')
        args.n_classes = None
        args.data_name = None
        args.total_epochs = None

    leaf_dir = args.class_method + '_pre_epoch_' + str(args.pre_epoch) + '_epoch_' + str(args.total_epochs) + '_seed_0'

    args.cls_dir = os.path.join('classifier_model', 'result', data_noise, leaf_dir) + '/'
    args.model_dir = os.path.join('classifier_model', 'result_model', data_noise,
                                  args.class_method) + '/pre_epoch_' + str(args.pre_epoch) + \
                     '_epoch_' + str(args.total_epochs) + '_dropout_ratio_' + str(args.dropout * 100) + '_seed_0' + '_'

    # classifier model
    os.makedirs(args.cls_dir, exist_ok=True)
    os.makedirs(os.path.join('classifier_model/result_model/', data_noise, args.class_method), exist_ok=True)

    # classifier model
    if args.class_method == 'no_stop':
        model = CE(args)
    elif args.class_method == 'joint':
        model = Joint(args)
    elif args.class_method =='coteaching':
        model = Coteaching(args)
    elif args.class_method =='jocor':
        model = JoCor(args)
    elif args.class_method == 'vanilla':
        model = ES(args)
    elif args.class_method == 'ls':
        model = LS(args)
    elif args.class_method == 'sce':
        model = SCE(args)
    elif args.class_method == 'rel':
        model = REL(args)
    elif args.class_method =='forward':
        model = Forward(args)
    elif args.class_method == 'dualt':
        model = DualT(args)
    elif args.class_method == 'tvr':
        model = TVR(args)
    elif args.class_method == 'causalnl':
        model = CausalNL(args)

    elif args.class_method == 'lrt':
        model = LRT(args)
    elif args.class_method == 'mlc':
        model = MLC(args)

    else:
        model = None

    pre_acc, train_class_acc, train_label_acc, test_acc, epoch = model.run()
    save_cls_text_configuration(args, pre_acc, train_class_acc, train_label_acc, test_acc, epoch)