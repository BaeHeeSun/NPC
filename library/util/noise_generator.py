import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from math import inf
from scipy import stats

# for noise generation
# Generate symmetric noise
def generate_dict_sym(loader, Noise_ratio, data_dict,is_Train=True, n_class = 10):
    Noise_ratio *= 10/9 # For MNIST/FMNIST/CIFAR10, n_class==10
    if is_Train:
        for i,(images, labels) in enumerate(tqdm(loader)):
            # 1. Choose train or valid
            if i<len(loader)*0.1:
                assign = 'Val_'
            else:
                assign = 'Train_'
            # 2. Choose clean or noisy
            if random.random() > Noise_ratio:  # noisy rate
                mode = 'Clean'
                new_label = labels  # clean == noisy
            else:
                mode = 'Noisy'
                new_label = torch.randint(low=0, high=n_class, size=(1,)) # generated label

            if len(data_dict[assign + mode]) == 0:
                data_dict[assign + mode]['image'] = images.numpy()
                data_dict[assign + mode]['label'] = new_label.numpy()
                data_dict[assign + mode]['class'] = labels.numpy()
            else:
                data_dict[assign + mode]['image'] = np.vstack((data_dict[assign + mode]["image"], images.numpy()))
                data_dict[assign + mode]['label'] = np.vstack((data_dict[assign + mode]["label"], new_label.numpy()))
                data_dict[assign + mode]['class'] = np.vstack((data_dict[assign + mode]['class'], labels.numpy()))

    else: # test
        assign = 'Test_'
        mode = 'Clean'
        for i, (images, labels) in enumerate(tqdm(loader)):
            if len(data_dict[assign + mode]) == 0:
                data_dict[assign + mode]['image'] = images.numpy()
                data_dict[assign + mode]['label'] = labels.numpy()
                data_dict[assign + mode]['class'] = labels.numpy()
            else:
                data_dict[assign + mode]['image'] = np.vstack((data_dict[assign + mode]["image"], images.numpy()))
                data_dict[assign + mode]['label'] = np.vstack((data_dict[assign + mode]["label"], labels.numpy()))
                data_dict[assign + mode]['class'] = np.vstack((data_dict[assign + mode]['class'], labels.numpy()))

    print(assign + 'Loader finished')
    return data_dict

# Generate asymmetric noise
def generate_dict_asym(loader, Noise_ratio, data_dict, label_list, is_Train=True):
    if is_Train:
        for i,(images, labels) in enumerate(tqdm(loader)):
            # 1. Choose train or valid
            if i < len(loader)*0.1:
                assign = 'Val_'
            else:
                assign = 'Train_'
            # 2. Choose clean or noisy
            if random.random() > Noise_ratio:  # noisy rate
                mode = 'Clean'
                new_label = labels  # clean == noisy
            else:
                mode = 'Noisy'
                new_label = torch.tensor(label_list[labels[0]])  # generated label

            if len(data_dict[assign + mode]) == 0:
                data_dict[assign + mode]['image'] = images.numpy()
                data_dict[assign + mode]['label'] = new_label.numpy()
                data_dict[assign + mode]['class'] = labels.numpy()
            else:
                data_dict[assign + mode]['image'] = np.vstack((data_dict[assign + mode]["image"], images.numpy()))
                data_dict[assign + mode]['label'] = np.vstack((data_dict[assign + mode]["label"], new_label.numpy()))
                data_dict[assign + mode]['class'] = np.vstack((data_dict[assign + mode]['class'], labels.numpy()))

    else:  # test
        assign = 'Test_'
        mode = 'Clean'
        for i, (images, labels) in enumerate(tqdm(loader)):
            if len(data_dict[assign + mode]) == 0:
                data_dict[assign + mode]['image'] = images.numpy()
                data_dict[assign + mode]['label'] = labels.numpy()
                data_dict[assign + mode]['class'] = labels.numpy()
            else:
                data_dict[assign + mode]['image'] = np.vstack((data_dict[assign + mode]["image"], images.numpy()))
                data_dict[assign + mode]['label'] = np.vstack((data_dict[assign + mode]["label"], labels.numpy()))
                data_dict[assign + mode]['class'] = np.vstack((data_dict[assign + mode]['class'], labels.numpy()))

    print(assign + 'Loader finished')
    return data_dict

# Generate idn noise
def generate_dict_idn(original_dict, Noise_ratio, data_dict, is_Train=True):
    if is_Train:
        data_to_see = original_dict['Train']
        data_len = len(data_to_see['l_true'])

        index = np.argsort(np.array(data_to_see['p_true']))
        num_noisy = int(data_len*Noise_ratio)
        noisy_index = index[:num_noisy]
        clean_index = index[num_noisy:]

        # clean
        np.random.shuffle(clean_index)
        valid_num = int(len(clean_index)*0.1)
        train_index = clean_index[valid_num:]
        valid_index = clean_index[:valid_num]

        data_dict['Train_Clean']['image'] = np.float32(np.array(data_to_see['images'])[train_index])
        data_dict['Train_Clean']['label'] = np.int64(np.array(data_to_see['l_true'])[train_index].reshape(-1,1))
        data_dict['Train_Clean']['class'] = np.int64(np.array(data_to_see['l_true'])[train_index].reshape(-1,1))

        data_dict['Val_Clean']['image'] = np.float32(np.array(data_to_see['images'])[valid_index])
        data_dict['Val_Clean']['label'] = np.int64(np.array(data_to_see['l_true'])[valid_index].reshape(-1,1))
        data_dict['Val_Clean']['class'] = np.int64(np.array(data_to_see['l_true'])[valid_index].reshape(-1,1))

        # noisy
        np.random.shuffle(noisy_index)
        valid_num = int(len(noisy_index)*0.1)
        train_index = noisy_index[valid_num:]
        valid_index = noisy_index[:valid_num]

        data_dict['Train_Noisy']['image'] = np.float32(np.array(data_to_see['images'])[train_index])
        data_dict['Train_Noisy']['label'] = np.int64(np.array(data_to_see['l_model'])[train_index].reshape(-1,1))
        data_dict['Train_Noisy']['class'] = np.int64(np.array(data_to_see['l_true'])[train_index].reshape(-1,1))

        data_dict['Val_Noisy']['image'] =np.float32( np.array(data_to_see['images'])[valid_index])
        data_dict['Val_Noisy']['label'] = np.int64(np.array(data_to_see['l_model'])[valid_index].reshape(-1,1))
        data_dict['Val_Noisy']['class'] = np.int64(np.array(data_to_see['l_true'])[valid_index].reshape(-1,1))

    else:
        data_to_see = original_dict['Test']

        data_dict['Test_Clean']['image'] = np.float32(np.array(data_to_see['images']))
        data_dict['Test_Clean']['label'] = np.int64(np.array(data_to_see['l_true']).reshape(-1,1))
        data_dict['Test_Clean']['class'] = np.int64(np.array(data_to_see['l_true']).reshape(-1,1))
    return data_dict

def generate_dict_idnx(loader, Noise_ratio, data_dict, feature_size, label_num, is_Train=True):
    flip_distribution = stats.truncnorm((0 - Noise_ratio) / 0.1, (1 - Noise_ratio) / 0.1, loc=Noise_ratio, scale=0.1)
    flip_rate = flip_distribution.rvs(len(loader))
    W = torch.randn(label_num, feature_size, label_num)

    if is_Train:
        for i,(images, labels) in enumerate(tqdm(loader)):
            # 1. Choose train or valid
            if i < len(loader)*0.1:
                assign = 'Val_'
            else:
                assign = 'Train_'
            p = images.view(1,-1).mm(W[labels].squeeze(0)).squeeze(0)
            p[labels] = -inf
            p = flip_rate[i]*torch.softmax(p, dim=0)
            p[labels]+=1-flip_rate[i]

            new_label = torch.multinomial(p,1)
            if labels==new_label:
                mode = 'Clean'
            else:
                mode = 'Noisy'

            if len(data_dict[assign + mode]) == 0:
                data_dict[assign + mode]['image'] = images.numpy()
                data_dict[assign + mode]['label'] = new_label.numpy()
                data_dict[assign + mode]['class'] = labels.numpy()
            else:
                data_dict[assign + mode]['image'] = np.vstack((data_dict[assign + mode]["image"], images.numpy()))
                data_dict[assign + mode]['label'] = np.vstack((data_dict[assign + mode]["label"], new_label.numpy()))
                data_dict[assign + mode]['class'] = np.vstack((data_dict[assign + mode]['class'], labels.numpy()))

    else:  # test
        assign = 'Test_'
        mode = 'Clean'
        for i, (images, labels) in enumerate(tqdm(loader)):
            if len(data_dict[assign + mode]) == 0:
                data_dict[assign + mode]['image'] = images.numpy()
                data_dict[assign + mode]['label'] = labels.numpy()
                data_dict[assign + mode]['class'] = labels.numpy()
            else:
                data_dict[assign + mode]['image'] = np.vstack((data_dict[assign + mode]["image"], images.numpy()))
                data_dict[assign + mode]['label'] = np.vstack((data_dict[assign + mode]["label"], labels.numpy()))
                data_dict[assign + mode]['class'] = np.vstack((data_dict[assign + mode]['class'], labels.numpy()))

    print(assign + 'Loader finished')
    return data_dict


def save_dict_to_pickle(data_name,Noise_ratio, total_data, noise_type):
    os.makedirs('data/',exist_ok=True)

    if Noise_ratio>0.0:
        noise_percent = str(Noise_ratio*100)
    else:
        noise_percent = '00.0'

    with open('data/'+data_name + "_" + noise_percent + '_' + noise_type+".pk", "wb") as f:
        pickle.dump(total_data, f)
    f.close()

