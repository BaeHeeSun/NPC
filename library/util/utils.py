import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from math import inf
from scipy import stats
from sklearn.manifold import TSNE

from dataloader import load_dataset

def plot_(dir, list, name):
    with open(dir + name +'.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(list)

    return

def from_model_to_dict(data_path, loader, length, device, net):
    labelarray = np.int64(np.zeros(length))
    for index, images, classes, _ in loader:
        images = images.to(device)
        if 'causalNL' in data_path:
            _, _, _, _, _, outputs, _ = net(images)
        else:
            _, outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        labelarray[index] = predicted.cpu().detach().numpy()

    return labelarray

def save_data(data_path, dataset, device, net):
    Ttloader, len_train, _ = dataset.train_test()
    trainloader= Ttloader['train']
    train_label = from_model_to_dict(data_path, trainloader, len_train, device, net)

    TVtloader, _, len_val, _ = dataset.train_val_test()
    validloader = TVtloader['valid']
    valid_label = from_model_to_dict(data_path, validloader, len_val, device, net)

    result_dict = {
        'Train': {'label': train_label}, 'Val_Clean': {'label': valid_label}}
    with open(data_path, "wb") as f:
        pickle.dump(result_dict, f)
    f.close()

    return

def save_cls_text_configuration(args, pre_acc, train_class_acc, train_label_acc, test_acc, epoch, post=False):
    if post:
        f = open(args.gen_dir + "Configs.txt", "w")
    else:
        f = open(args.cls_dir + "Configs.txt", "w")
    for arg in vars(args):
        f.write("{} : {}".format(arg, getattr(args, arg)))
        f.write('\n')
    f.write('======================================')
    f.write('\n')
    f.write("pre acc : " + str(pre_acc))
    f.write('\n')
    f.write("train class acc : " + str(train_class_acc))
    f.write('\n')
    f.write("train label acc : " + str(train_label_acc))
    f.write('\n')
    f.write("test acc : " + str(test_acc))
    f.write('\n')
    f.write("stopping epoch : " + str(epoch))
    f.write('\n')
    f.close()

    return

def save_gen_text_configuration(args, pre_loss, train_loss, train_recon_loss, test_loss, test_acc):
    f = open(args.gen_dir + "Configs.txt", "w")
    for arg in vars(args):
        f.write("{} : {}".format(arg, getattr(args, arg)))
        f.write('\n')
    f.write('======================================')
    f.write('\n')
    f.write("pre loss : " + str(pre_loss))
    f.write('\n')
    f.write("train loss : " + str(train_loss))
    f.write('\n')
    f.write("train recon loss : " + str(train_recon_loss))
    f.write('\n')
    f.write("test loss : " + str(test_loss))
    f.write('\n')
    f.write("test acc : " + str(test_acc))
    f.write('\n')
    f.close()

    return

def save_gen_text_configuration_v2(args, test_acc):

    f = open(args.gen_dir + "Configs.txt", "w")
    for arg in vars(args):
        f.write("{} : {}".format(arg, getattr(args, arg)))
        f.write('\n')
    f.write('======================================')
    f.write('\n')
    f.write("test acc : " + str(test_acc))
    f.write('\n')
    f.close()

    return

