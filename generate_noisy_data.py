import torchvision
import torchvision.transforms as transforms

from library.util.noise_generator import *

####################################################################################################################
### MNIST_28*28
####################################################################################################################

# Define Transform
transform = transforms.Compose([transforms.ToTensor()])

# Load Data
train_dataset = torchvision.datasets.MNIST(root='./', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./', train=False, transform=transform)
train_loader, test_loader = DataLoader(dataset=train_dataset), DataLoader(dataset=test_dataset)

# Sym
for n_ratio in [0.2, 0.5, 0.8]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean':dict(), 'Val_Noisy':dict(), "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_sym(train_loader, n_ratio, total_data)
    total_data = generate_dict_sym(test_loader, n_ratio, total_data, is_Train=False)
    save_dict_to_pickle('MNIST', n_ratio, total_data, 'sym')
    print('MNIST symmetric noise', n_ratio)

# Asym
for Noise_ratio in [0.2, 0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean': dict(), 'Val_Noisy': dict(),
                  "Test_Clean": dict(), "Test_Noisy": dict()}
    label_list = [0, 1, 7, 8, 4, 6, 5, 7, 8, 9]  # from iclr 2021 REL
    total_data = generate_dict_asym(train_loader, Noise_ratio, total_data, label_list)
    total_data = generate_dict_asym(test_loader, Noise_ratio, total_data, label_list, is_Train=False)
    save_dict_to_pickle('MNIST', Noise_ratio, total_data, 'asym')
    print('MNIST asymmetric noise1', Noise_ratio)

# IDN
for Noise_ratio in [0.2,0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean': dict(), 'Val_Noisy': dict(),
                  "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_idnx(train_loader, Noise_ratio, total_data, 1*28*28, 10)
    total_data = generate_dict_idnx(test_loader, Noise_ratio, total_data, 1 * 28 * 28, 10, is_Train=False)
    save_dict_to_pickle('MNIST', Noise_ratio, total_data, 'idnx')
    print('MNIST IDNX noise', Noise_ratio)

# SRIDN
with open('idn_data/MNIST.pk', 'rb') as f:
    data_dict = pickle.load(f)

for n_ratio in [0.1,0.2,0.3,0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean': dict(), 'Val_Noisy': dict(), "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_idn(data_dict, n_ratio, total_data)
    total_data = generate_dict_idn(data_dict, n_ratio, total_data, is_Train=False)
    save_dict_to_pickle('MNIST', n_ratio, total_data, 'idn')
    print('MNIST IDN Noise', n_ratio)

####################################################################################################################
### FMNIST_28*28
####################################################################################################################

# Define Transform
transform = transforms.Compose([transforms.ToTensor()])

# Load Data
train_dataset = torchvision.datasets.FashionMNIST(root='./', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./', train=False, transform=transform)
train_loader, test_loader = DataLoader(dataset=train_dataset), DataLoader(dataset=test_dataset)

# Sym
for n_ratio in [0.2, 0.5, 0.8]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean':dict(), 'Val_Noisy':dict(), "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_sym(train_loader, n_ratio, total_data)
    total_data = generate_dict_sym(test_loader, n_ratio, total_data, is_Train=False)
    save_dict_to_pickle('FMNIST', n_ratio, total_data, 'sym')
    print('FMNIST symmetric noise', n_ratio)

# Asym
for Noise_ratio in [0.2, 0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean': dict(), 'Val_Noisy': dict(),
                  "Test_Clean": dict(), "Test_Noisy": dict()}
    label_list = [6, 1, 4, 3, 4, 7, 6, 7, 8, 9]
    total_data = generate_dict_asym(train_loader, Noise_ratio, total_data, label_list)
    total_data = generate_dict_asym(test_loader, Noise_ratio, total_data, label_list, is_Train=False)
    save_dict_to_pickle('FMNIST', Noise_ratio, total_data, 'asym')
    print('FMNIST asymmetric noise', Noise_ratio)

# IDN
for Noise_ratio in [0.2,0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean': dict(), 'Val_Noisy': dict(),
                  "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_idnx(train_loader, Noise_ratio, total_data, 1*28*28, 10)
    total_data = generate_dict_idnx(test_loader, Noise_ratio, total_data, 1 * 28 * 28, 10, is_Train=False)
    save_dict_to_pickle('FMNIST', Noise_ratio, total_data, 'idnx')
    print('FMNIST IDNX noise', Noise_ratio)

# SRIDN
with open('idn_data/FMNIST.pk', 'rb') as f:
    data_dict = pickle.load(f)

for n_ratio in [0.1,0.2,0.3,0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean': dict(), 'Val_Noisy': dict(), "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_idn(data_dict, n_ratio, total_data)
    total_data = generate_dict_idn(data_dict, n_ratio, total_data, is_Train=False)
    save_dict_to_pickle('FMNIST', n_ratio, total_data, 'idn')
    print('FMNIST IDN Noise', n_ratio)

####################################################################################################################
### CIFAR10 dataset
####################################################################################################################

# Define Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load Data
trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)

trainloader, testloader = DataLoader(dataset=trainset), DataLoader(dataset=testset)

# Sym
for Noise_ratio in [0.2,0.5,0.8]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean':dict(), 'Val_Noisy':dict(), "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_sym(trainloader,Noise_ratio, total_data)
    total_data = generate_dict_sym(testloader,Noise_ratio, total_data, is_Train=False)
    save_dict_to_pickle('CIFAR10',Noise_ratio, total_data, 'sym')
    print('CIFAR10 symmetric noise',Noise_ratio)

#[plane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
# Asym
for Noise_ratio in [0.2, 0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean':dict(), 'Val_Noisy':dict(), "Test_Clean": dict(), "Test_Noisy": dict()}
    label_list = [0,1,0,5,7,3,6,7,8,1]
    total_data = generate_dict_asym(trainloader,Noise_ratio,total_data, label_list)
    total_data = generate_dict_asym(testloader,Noise_ratio,total_data, label_list, is_Train=False)
    save_dict_to_pickle('CIFAR10',Noise_ratio, total_data, 'asym')
    print('CIFAR10 asymmetric noise',Noise_ratio)

# IDN
for Noise_ratio in [0.2,0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean': dict(), 'Val_Noisy': dict(),
                  "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_idnx(trainloader, Noise_ratio, total_data, 3*32*32, 10)
    total_data = generate_dict_idnx(testloader, Noise_ratio, total_data, 3*32*32, 10, is_Train=False)
    save_dict_to_pickle('CIFAR10', Noise_ratio, total_data, 'idnx')
    print('CIFAR10 IDNX noise', Noise_ratio)

# SRIDN
with open('idn_data/CIFAR10.pk', 'rb') as f:
    data_dict = pickle.load(f)

for n_ratio in [0.1,0.2,0.3,0.4]:
    total_data = {"Train_Clean": dict(), "Train_Noisy": dict(), 'Val_Clean': dict(), 'Val_Noisy': dict(), "Test_Clean": dict(), "Test_Noisy": dict()}
    total_data = generate_dict_idn(data_dict, n_ratio, total_data)
    total_data = generate_dict_idn(data_dict, n_ratio, total_data, is_Train=False)
    save_dict_to_pickle('CIFAR10', n_ratio, total_data, 'idn')
    print('CIFAR10 IDN Noise', n_ratio)

####################################################################################################################

