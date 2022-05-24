import argparse
import os
import numpy as np
import torch

####################################################################################################################
parser = argparse.ArgumentParser(description="main")

# data
parser.add_argument('--dataset', type=str, default='MNIST', help = 'MNIST, FMNIST, CIFAR10, Clothing, Food')
parser.add_argument('--noise_type', type=str, default='clean', help='clean, sym, asym, idn, idnx')
parser.add_argument('--noisy_ratio', type=float, default=0.4, help='between 0 and 1')

# classifier
parser.add_argument('--class_method', type=str, default=None)
parser.add_argument('--post_method', type=str, default=None)

# prior mode
parser.add_argument('--knn_mode', type=str, default=None, help='onehot, proba')
parser.add_argument('--selected_class', type=str, default='1',help='2,5,10, ...n_class')

# experiment condition for generator
parser.add_argument('--prior_norm', type=float, default=5, help='rho')
parser.add_argument('--beta', type=float, default=1.0, help='coefficient on kl loss, beta vae')
parser.add_argument("--total_iter", type=int, default=10, help='total iter (Default : 10)')

# general experiment condition
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--lr", type=float, default=0.001, help = "Learning rate (Default : 1e-3)")
parser.add_argument('--softplus_beta', type=float, default=1, help='softplus beta')
parser.add_argument('--clip_gradient_norm', type=float, default=100000, help='max norm for gradient clipping')

# etc
parser.add_argument('--set_gpu', type=int, default=0, help='gpu setting 0/1/2/3')
parser.add_argument('--data_dir', type=str, default=None)

####################################################################################################################
if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.set_gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data and network
    if args.dataset in ['MNIST','FMNIST','CIFAR10']:
        args.n_classes = 10
        args.dropout = 0.3
        args.causalnl_z_dim = 25
        args.batch_size = 128
        if args.dataset in ['MNIST', 'FMNIST']:
            args.model = 'CNN_MNIST'
        elif args.dataset == 'CIFAR10':
            args.model = 'CNN_CIFAR'
    elif args.dataset in ['Clothing', 'Food']:
        if args.dataset == 'Clothing':
            args.n_classes = 14
        elif args.dataset == 'Food':
            args.n_classes = 101
        args.noise_type = 'clean'
        args.dropout = 0.0
        args.causalnl_z_dim = 100
        args.batch_size = 32
        args.model = 'Resnet50Pre'
    else: # wrong dataset
        args.n_classes = None
        args.dropout = None
        args.batch_size = None

    # classifier model
    if args.dataset in ['MNIST', 'FMNIST', 'CIFAR10']:
        args.pre_epoch = 10
        if args.class_method in ['cores', 'rel'] :
            args.total_epochs = 100
        elif args.class_method =='causalNL':
            args.total_epochs = 150
            args.dropout = 0.0
        else:
            args.total_epochs = 200
    elif args.dataset in ['Food']:
        args.pre_epoch  = 5
        if args.class_method in ['cores', 'rel']:
            args.total_epochs = 50
        else:
            args.total_epochs = 100
    else: # Clothing
        args.pre_epoch = 1
        args.total_epochs = 10

    # directory
    if args.noise_type == 'clean':
        args.data_name = args.dataset + '_00.0_' + args.noise_type + '.pk'
        data_noise = args.dataset + '_' + args.model + '_clean'
    else:
        args.data_name = args.dataset + '_' + str(100 * args.noisy_ratio) + '_' + args.noise_type + '.pk'
        data_noise = args.dataset + '_' + args.model + '_' + args.noise_type + '_' + str(100 * args.noisy_ratio)

    if args.dataset in ['Clothing', 'Food']:
        args.data_name = args.dataset

    # directory for loading trained classifier
    data_leaf = args.class_method+'_pre_epoch_'+str(args.pre_epoch)+'_epoch_'+str(args.total_epochs)+'_seed_0'

    args.cls_dir = os.path.join('classifier_model', 'result', data_noise, data_leaf) + '/'
    args.model_dir = os.path.join('classifier_model','result_model',data_noise,args.class_method)\
                     +'/pre_epoch_'+str(args.pre_epoch)+ '_epoch_'+str(args.total_epochs)+'_dropout_ratio_'+str(args.dropout * 100)+'_seed_0_'

    # post model
    if args.post_method == 'rog':
        from library.rog import ROG
        model = ROG(args)
        acc = model.run()

        cls_dir = os.path.join('classifier_model', 'result', data_noise,
                               'rog_pre_epoch_' + str(args.pre_epoch) + '_epoch_' + str(args.total_epochs) +
                               '_seed_' + str(args.seed) + '_pre_method_' + str(args.class_method)) + '/'
        os.makedirs(cls_dir, exist_ok=True)

        f = open(cls_dir + "_Acc.txt", "w")
        f.write('======================================')
        f.write('\n')
        f.write("test acc : " + str(acc))
        f.write('\n')
        f.close()

    elif args.post_method == 'knn':
        from library.knn_test import KNN_tester
        func = KNN_tester(args)
        func.knn_test()

    else:
        from train_npc import NPC
        from evaluate import Acc_calculator
        from library.util.utils import save_gen_text_configuration_v2

        if args.knn_mode == 'onehot':
            args.selected_class = '1'
        # generate directory to save npc result
        args.gen_dir = os.path.join('result', data_noise, data_leaf) \
                       + '/beta_' + str(args.beta) \
                       + '_prior_norm_' + str(args.prior_norm) \
                       + '_gen_epoch_' + str(args.total_iter) \
                       + '_seed_' + str(args.seed) \
                       + '_act_relu_clip_' + str(args.clip_gradient_norm) \
                       + '_softp_' + str(args.softplus_beta) \
                       + '/' + args.knn_mode + '_' + args.selected_class + '/'
        args.gen_model_dir = os.path.join('result_model', data_noise, args.class_method) \
                             + '/beta_' + str(args.beta) \
                             + '_prior_norm_' + str(args.prior_norm) \
                             + '_gen_epoch_' + str(args.total_iter) \
                             + '_seed_' + str(args.seed) \
                             + '_act_relu_clip_' + str(args.clip_gradient_norm) \
                             + '_softp_' + str(args.softplus_beta) \
                             + '_knn_' + args.knn_mode + '_' + args.selected_class

        # npc directory
        os.makedirs(args.gen_dir, exist_ok=True)
        os.makedirs(os.path.join('result_model', data_noise, args.class_method), exist_ok=True)

        scaler = torch.cuda.amp.GradScaler()
        gen_model = NPC(args, scaler)
        gen_model.run()

        # model performance calculation process
        func = Acc_calculator(args)
        accuracy = func.merge_classifier_and_autoencoder()
        save_gen_text_configuration_v2(args, accuracy)
