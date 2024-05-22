import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

def get_dataset(args):
    if args.dataset == "MNIST":

        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])  
        train_dataset_raw = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_dataset_raw, [0.9, 0.1])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize * 8, shuffle=False)
        return train_loader, val_loader, test_loader
    
    if args.dataset == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.2860), (0.3530))
                                    ])  

        train_dataset_raw = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_dataset_raw, [0.9, 0.1])
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize * 8, shuffle=False)
        return train_loader, val_loader, test_loader
    
    if args.dataset == 'CIFAR10':
        # transform with 
        # mean [0.49139968 0.48215841 0.44653091]
        # stdd [0.24703223 0.24348513 0.26158784]
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
                                    ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        return train_loader, val_loader, test_loader
    

    if args.dataset == 'CIFAR100':
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
                                    ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        return train_loader, val_loader, test_loader

# compute the mean and std of the dataset
# import torch
# from torchvision import datasets, transforms

# print("MNIST")
# train_dataset_raw = datasets.MNIST(root='./data', train=True, download=True)
# print(train_dataset_raw.train_data.shape)
# print(train_dataset_raw.train_data.float().mean(axis=(0,1,2))/255)
# print(train_dataset_raw.train_data.float().std(axis=(0,1,2))/255)

# print("\n FashionMNIST")
# train_dataset_raw = datasets.FashionMNIST(root='./data', train=True, download=True)
# print(train_dataset_raw.train_data.shape)
# print(train_dataset_raw.train_data.float().mean(axis=(0,1,2))/255)
# print(train_dataset_raw.train_data.float().std(axis=(0,1,2))/255)

# print("\n CIFAR10")
# train_dataset_raw = datasets.CIFAR10(root='./data', train=True, download=True)
# print(train_dataset_raw.data.shape)
# print(train_dataset_raw.data.mean(axis=(0,1,2))/255)
# print(train_dataset_raw.data.std(axis=(0,1,2))/255)

# print("\n CIFAR100")
# train_dataset_raw = datasets.CIFAR100(root='./data', train=True, download=True)
# print(train_dataset_raw.data.shape)
# print(train_dataset_raw.data.mean(axis=(0,1,2))/255)
# print(train_dataset_raw.data.std(axis=(0,1,2))/255)