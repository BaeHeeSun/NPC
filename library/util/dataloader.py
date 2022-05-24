from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import os
from PIL import Image
import pickle
import numpy as np

class DataIterator(object): # for meta learning
    def __init__(self, dataloader):
        self.loader = dataloader
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            idx, x, y, y_tilde = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx, x, y, y_tilde = next(self.iterator)

        return idx, x, y, y_tilde

class revised_dataset(Dataset): # mnist/fminst/cifar 10
    def __init__(self, filename, mode, root, validation=False):
        with open(root+filename, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.mode = mode

        self.noise_ratio = float(filename.split('_')[1])

        if self.noise_ratio <= 1: # cleanset
            if validation: # valid mode
                self.data_train_image = self.data['Train_Clean']['image']
                self.data_train_class = self.data['Train_Clean']['class']
                self.data_train_label = self.data['Train_Clean']['label']

                self.data_val_image = self.data['Val_Clean']['image']
                self.data_val_class = self.data['Val_Clean']['class']
                self.data_val_label = self.data['Val_Clean']['label']

            else: # not validation
                self.data_train_image = np.vstack((self.data['Train_Clean']['image'], self.data['Val_Clean']['image']))
                self.data_train_class = np.vstack((self.data['Train_Clean']['class'], self.data['Val_Clean']['class']))
                self.data_train_label = np.vstack((self.data['Train_Clean']['label'], self.data['Val_Clean']['label']))

        elif self.noise_ratio>1: # noisyset
            if validation:
                self.data_train_image = np.vstack((self.data['Train_Clean']['image'], self.data['Train_Noisy']['image']))
                self.data_train_class = np.vstack((self.data['Train_Clean']['class'], self.data['Train_Noisy']['class']))
                self.data_train_label = np.vstack((self.data['Train_Clean']['label'], self.data['Train_Noisy']['label']))

                self.data_val_image = np.vstack((self.data['Val_Clean']['image'], self.data['Val_Noisy']['image']))
                self.data_val_class = np.vstack((self.data['Val_Clean']['class'], self.data['Val_Noisy']['class']))
                self.data_val_label = np.vstack((self.data['Val_Clean']['label'], self.data['Val_Noisy']['label']))

            else:
                self.data_train_image = np.vstack((self.data['Train_Clean']['image'],self.data['Train_Noisy']['image'],self.data['Val_Clean']['image'],self.data['Val_Noisy']['image']))
                self.data_train_class = np.vstack((self.data['Train_Clean']['class'],self.data['Train_Noisy']['class'],self.data['Val_Clean']['class'],self.data['Val_Noisy']['class']))
                self.data_train_label = np.vstack((self.data['Train_Clean']['label'],self.data['Train_Noisy']['label'],self.data['Val_Clean']['label'],self.data['Val_Noisy']['label']))

        self.data_test_image = self.data['Test_Clean']['image']
        self.data_test_class = self.data['Test_Clean']['class']
        self.data_test_label = self.data['Test_Clean']['label']

    def __getitem__(self, item):
        # train/test split
        if self.mode == 'Train':
            img = self.data_train_image[item]
            class_ = self.data_train_class[item][0]
            label = self.data_train_label[item][0]
            return item, img, class_, label

        elif self.mode == 'Valid':
            img = self.data_val_image[item]
            class_ = self.data_val_class[item][0]
            label = self.data_val_label[item][0]
            return item, img, class_, label

        elif self.mode == 'Test':
            img = self.data_test_image[item]
            class_ = self.data_test_class[item][0]
            label = self.data_test_label[item][0]
            return item, img, class_, label

    def __len__(self):
        if self.mode == 'Train':
            return len(self.data_train_image)
        elif self.mode == 'Valid':
            return len(self.data_val_image)
        elif self.mode == 'Test':
            return len(self.data_test_image)

class clothing_dataset(Dataset): # clothing1m
    def __init__(self, mode, root, validation=False):
        self.mode = mode
        self.root = root+'Clothing_1M'

        # define transform
        if self.mode == 'Test':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])
        else: # train/validation
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])

        self.train_images, self.train_labels = [], []
        self.test_images, self.test_labels = [], []
        if validation: # train/valid/test split
            self.val_images, self.val_labels = [], []
            with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    criteria = random.random()
                    entry = l.split()
                    img_path = entry[0][7:]
                    if criteria>0.9:
                        self.val_images.append(img_path)
                        self.val_labels.append(int(entry[1]))
                    else:
                        self.train_images.append(img_path)
                        self.train_labels.append(int(entry[1]))
            with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0][7:]
                    self.test_images.append(img_path)
                    self.test_labels.append(int(entry[1]))

        else: # train/test split
            with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0][7:]
                    self.train_images.append(img_path)
                    self.train_labels.append(int(entry[1]))
            with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0][7:]
                    self.test_images.append(img_path)
                    self.test_labels.append(int(entry[1]))

    def __getitem__(self, item):
        if self.mode == 'Train':
            img_path = self.train_images[item]
            label = self.train_labels[item]
            image = Image.open(self.root+'/'+img_path).convert('RGB')
            img = self.transform(image)

            return item, img, label, label

        elif self.mode == 'Valid':
            img_path = self.val_images[item]
            label = self.val_labels[item]
            image = Image.open(self.root + '/' + img_path).convert('RGB')
            img = self.transform(image)

            return item, img, label, label

        elif self.mode == 'Test':
            img_path = self.test_images[item]
            label = self.test_labels[item]
            image = Image.open(self.root + '/' + img_path).convert('RGB')
            img = self.transform(image)

            return item, img, label, label

    def __len__(self):
        if self.mode == 'Train':
            return len(self.train_labels)
        elif self.mode == 'Valid':
            return len(self.val_labels)
        elif self.mode == 'Test':
            return len(self.test_labels)


class food_dataset(Dataset): # food101
    def __init__(self, mode, root, validation=False):
        self.mode = mode
        self.root = root+'FOOD_101M/meta'
        self.img_root = root+'FOOD_101M/images'

        # define classes
        with open(self.root+'/classes.txt', 'r') as f:
            self.classes = f.read().splitlines()

        # define transform
        if self.mode == 'Test':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else: # train/validation
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        self.train_images, self.train_labels = [], []
        self.test_images, self.test_labels = [], []
        if validation: # train/valid/test split
            self.val_images, self.val_labels = [], []
            with open(self.root+'/train.txt', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    criteria = random.random()
                    img_name = l
                    entry = l.split('/')
                    label = self.classes.index(entry[0])
                    if criteria>0.9:
                        self.val_images.append(img_name)
                        self.val_labels.append(label)
                    else:
                        self.train_images.append(img_name)
                        self.train_labels.append(label)
            with open(self.root+'/test.txt', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_name = l
                    entry = l.split('/')
                    label = self.classes.index(entry[0])
                    self.test_images.append(img_name)
                    self.test_labels.append(label)

        else: # train/test split
            with open(self.root + '/train.txt', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_name = l
                    entry = l.split('/')
                    label = self.classes.index(entry[0])
                    self.train_images.append(img_name)
                    self.train_labels.append(label)
            with open(self.root + '/test.txt', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_name = l
                    entry = l.split('/')
                    label = self.classes.index(entry[0])
                    self.test_images.append(img_name)
                    self.test_labels.append(label)

    def __getitem__(self, item):
        if self.mode == 'Train':
            img_path = self.train_images[item]
            label = self.train_labels[item]
            image = Image.open(self.img_root+'/'+img_path+'.jpg').convert('RGB')
            img = self.transform(image)

            return item, img, label, label

        elif self.mode == 'Valid':
            img_path = self.val_images[item]
            label = self.val_labels[item]
            image = Image.open(self.img_root + '/' + img_path+'.jpg').convert('RGB')
            img = self.transform(image)

            return item, img, label, label

        elif self.mode == 'Test':
            img_path = self.test_images[item]
            label = self.test_labels[item]
            image = Image.open(self.img_root + '/' + img_path+'.jpg').convert('RGB')
            img = self.transform(image)

            return item, img, label, label

    def __len__(self):
        if self.mode == 'Train':
            return len(self.train_labels)
        elif self.mode == 'Valid':
            return len(self.val_labels)
        elif self.mode == 'Test':
            return len(self.test_labels)

class load_dataset():
    def __init__(self,filename,batch_size, dir='data/'):
        self.filename = filename
        self.batch_size = batch_size
        self.dir = dir

    def train_test(self): # train/test
        if '.pk' in self.filename: # MNIST/CIFAR
            self.data_train = revised_dataset(self.filename, 'Train', root=self.dir, validation=False)
            self.data_test = revised_dataset(self.filename, 'Test', root=self.dir, validation=False)
        elif self.filename=='Food':
            self.data_train = food_dataset('Train', root=self.dir, validation=False)
            self.data_test = food_dataset('Test', root=self.dir, validation=False)
        elif self.filename=='Clothing':
            self.data_train = clothing_dataset('Train', root=self.dir, validation=False)
            self.data_test = clothing_dataset('Test', root=self.dir, validation=False)

        dataloaders = {}
        dataloaders['train'] = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        dataloaders['test'] = DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=8)

        return dataloaders, len(self.data_train), len(self.data_test)

    def train_val_test(self): # train/valid/test
        validation=True
        if '.pk' in self.filename:
            self.data_train = revised_dataset(self.filename, 'Train', root=self.dir, validation=validation)
            self.data_val = revised_dataset(self.filename, 'Valid', root=self.dir, validation=validation)
            self.data_test = revised_dataset(self.filename, 'Test', root=self.dir, validation=validation)
        elif self.filename=='Food':
            self.data_train = food_dataset('Train', root=self.dir, validation=validation)
            self.data_val = food_dataset('Valid', root=self.dir, validation=validation)
            self.data_test = food_dataset('Test', root=self.dir, validation=validation)
        elif self.filename=='Clothing':
            self.data_train = clothing_dataset('Train', root=self.dir, validation=validation)
            self.data_val = clothing_dataset('Valid', root=self.dir, validation=validation)
            self.data_test = clothing_dataset('Test', root=self.dir, validation=validation)

        dataloaders = {}
        dataloaders['train'] = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True,num_workers=8)
        dataloaders['valid'] = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=True, pin_memory=True,num_workers=8)
        dataloaders['test'] = DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=8)

        return dataloaders, len(self.data_train), len(self.data_val), len(self.data_test)

    def meta_train_val_test(self):
        validation = True
        if '.pk' in self.filename:
            self.data_meta = revised_dataset(self.filename, 'Valid', root=self.dir, validation=validation)
            self.data_train = revised_dataset(self.filename, 'Train', root=self.dir, validation=validation)
            self.data_val = revised_dataset(self.filename, 'Valid', root=self.dir, validation=validation)
            self.data_test = revised_dataset(self.filename, 'Test', root=self.dir, validation=validation)
        elif self.filename == 'Food':
            self.data_meta = food_dataset('Valid', root=self.dir, validation=validation)
            self.data_train = food_dataset('Train', root=self.dir, validation=validation)
            self.data_val = food_dataset('Valid', root=self.dir, validation=validation)
            self.data_test = food_dataset('Test', root=self.dir, validation=validation)
        elif self.filename == 'Clothing':
            self.data_meta = clothing_dataset('Valid', root=self.dir, validation=validation)
            self.data_train = clothing_dataset('Train', root=self.dir, validation=validation)
            self.data_val = clothing_dataset('Valid', root=self.dir, validation=validation)
            self.data_test = clothing_dataset('Test', root=self.dir, validation=validation)

        dataloaders = {}
        dataloaders['meta'] = DataIterator(DataLoader(self.data_meta, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8))
        dataloaders['train'] = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        dataloaders['valid'] = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        dataloaders['test'] = DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=8)

        return dataloaders, len(self.data_meta), len(self.data_train), len(self.data_val), len(self.data_test)

