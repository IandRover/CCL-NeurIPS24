import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_dataset(args):
    if args.dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), 
                                       ])
        train_dataset_raw = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_dataset_raw, [0.9, 0.1])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize * 8, shuffle=False)
        return train_loader, val_loader, test_loader
    
    if args.dataset == "MNIST_CNN":
        
        print(f"Using Randomcrop, horizontalflip")

        # Define the transformations
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(32),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
        ])

        # Load the MNIST dataset without any transformations
        train_dataset_raw = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        train_dataset_raw, val_dataset_raw = random_split(train_dataset_raw, [0.9, 0.1])

        # Apply transformations to the training dataset
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        train_dataset.data = train_dataset_raw.dataset.data[train_dataset_raw.indices]
        train_dataset.targets = train_dataset_raw.dataset.targets[train_dataset_raw.indices]

        # Apply transformations to the validation dataset
        val_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=test_transform)
        val_dataset.data = val_dataset_raw.dataset.data[val_dataset_raw.indices]
        val_dataset.targets = val_dataset_raw.dataset.targets[val_dataset_raw.indices]
        
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    if args.dataset == "STL10":
        print(f"Using Randomcrop, horizontalflip")

        # Define the transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load the STL-10 dataset without any transformations
        train_dataset_raw = datasets.STL10(root='./data', split='train', download=False, transform=None)
        train_dataset_raw, val_dataset_raw = random_split(train_dataset_raw, [0.9, 0.1])

        # Apply transformations to the training dataset
        train_dataset = datasets.STL10(root='./data', split='train', download=False, transform=train_transform)
        train_dataset.data = train_dataset_raw.dataset.data[train_dataset_raw.indices]
        train_dataset.labels = train_dataset_raw.dataset.labels[train_dataset_raw.indices]

        # Apply transformations to the validation dataset
        val_dataset = datasets.STL10(root='./data', split='train', download=False, transform=test_transform)
        val_dataset.data = val_dataset_raw.dataset.data[val_dataset_raw.indices]
        val_dataset.labels = val_dataset_raw.dataset.labels[val_dataset_raw.indices]

        test_dataset = datasets.STL10(root='./data', split='test', download=False, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

        return train_loader, val_loader, test_loader
    
    if args.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4, padding_mode='edge'),
                                                normalize,
                                        ])
        test_transform = transforms.Compose([transforms.ToTensor(), 
                                                normalize
                                            ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        return train_loader, val_loader, test_loader
    

    if args.dataset == 'OxfordIIITPet':
        train_transform = val_transform = target_transform = transforms.Compose([ transforms.ToTensor(),
                                                                                #   transforms.CenterCrop(128),
                                                                                 transforms.Resize((128, 128)),
                                                                            ])

        train_dataset = datasets.OxfordIIITPet(root='./data/OxfordIIITPet', split='trainval', download=False, target_types='segmentation',
                                                transform=train_transform, target_transform=target_transform)

        test_dataset = datasets.OxfordIIITPet(root='./data/OxfordIIITPet', split='test', download=False, target_types='segmentation',
                                            transform=val_transform, target_transform=target_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        return train_loader, test_loader, test_loader  # No separate test dataset for Cityscapes
    

def gradient_centralization(model):
    with torch.no_grad():
        for p1, p2 in model.named_parameters():
            if "bias" in p1 or p2.grad is None: continue
            if len(p2.shape) == 2: p2.grad -= p2.grad.mean(dim=1,keepdim=True)
            elif len(p2.shape) == 4: p2.grad -= p2.grad.mean(dim=[1,2,3],keepdim=True) 


def get_activation_function(func_name):
    if func_name == "tanh": return F.tanh
    elif func_name == "elu": return F.elu
    elif func_name == "relu": return F.relu
    elif func_name == "leaky_relu": return F.leaky_relu
    elif func_name == "selu": return F.selu
    else: raise ValueError(f"Activation function {func_name} not implemented")

def normalize_along_axis(x):
    x = x.reshape(len(x), -1)
    norm = torch.norm(x, dim=1, keepdim=True)
    return x / (norm + 1e-8)


class SigmaLoss(nn.Module):
    def __init__(self, args):
        super(SigmaLoss, self).__init__()
        self.args = args
        self.final_criteria = nn.CrossEntropyLoss()
        self.local_criteria = compute_SCL_loss
        self.method = args.method

        self.max_pool = nn.MaxPool2d(2, 2)
        self.max_pool = nn.AvgPool2d(2, 2)
        self.embed_target = args.embed_target

    def forward(self, activations, signals, target, method="final"):
        if method == "local":
            loss = list()
            for idx, (act, sig) in enumerate(zip(activations[:-1], signals[:-1])):
                if len(act.shape) == 4 and len(sig.shape) == 2: sig = sig.view(sig.shape[0], sig.shape[1], act.shape[2], act.shape[3]) 
                if len(act.shape) == 2 and len(sig.shape) == 4: act = act.view(act.shape[0], act.shape[1], sig.shape[2], sig.shape[3])
                if self.embed_target == 1 and act.shape[1] % 4 != 0:
                    target_layer = torch.zeros((10, 10, act.shape[2], act.shape[3])).to(act.device)
                    target_layer[np.arange(10),np.arange(10)] = 1
                    sig = torch.cat([sig, target_layer], dim=1)
                loss += [self.local_criteria(self.args, act, sig, target, predictions=activations[-1])]
            loss += [self.final_criteria(activations[-1], target)]
            return sum(loss), loss[-1].item()
        elif method == "final":
            loss = self.final_criteria(activations[-1], target)
            return loss, loss.item()
        
def compute_SCL_loss(args, A, B, target, predictions):

    if args.filter_target != 0:
        with torch.no_grad():
            softmax_output = F.softmax(predictions, dim=1)
            target_temp = torch.zeros_like(softmax_output, dtype=torch.float32)
            target_temp.scatter_(1, target.unsqueeze(1), 1.0)
            diff = torch.abs(softmax_output - target_temp)
            mask = 1 - torch.sum((diff < 0.1) * target_temp, dim=1).to(torch.float32)
        if len(A.shape) == 2:
            A = A * mask.unsqueeze(1)
        elif len(A.shape) == 4:
            A = A * mask.view(mask.shape[0], 1, 1, 1)
        else:
            raise ValueError(f"Unsupported shape for A: {A.shape}")

    A_norm, B_norm = normalize_along_axis(A), normalize_along_axis(B)
    
    C = A_norm@B_norm.T
    
    if len(B) == 10:
        identity = (target.unsqueeze(1) == args.T10).float().to(target.device)
    else:
        target_A = torch.zeros_like(C, dtype=torch.float32, device=target.device)
        target_A.scatter_(1, target.unsqueeze(1), 1.0)
        identity = torch.matmul(target_A, target_A.T)

    D = torch.matmul(A_norm, A_norm.T)
    identity_D = torch.eye(A.shape[0]).to(A.device)

    E = torch.matmul(B_norm, B_norm.T)
    identity_E = torch.eye(B.shape[0]).to(B.device)
    
    if args.loss_scale_C == 0:
        loss_C = F.mse_loss(C, identity)
    else:
        loss_C = F.mse_loss(C, identity) * A.shape[2] / args.loss_scale_C
    loss_D = F.mse_loss(D, identity_D) * args.loss_scale_A
    loss_E = F.mse_loss(E, identity_E) * args.loss_scale_B

    return loss_C + loss_D + loss_E

def compute_SCL_loss_AE(args, A, B, target, predictions):

    assert A.shape == B.shape

    A_norm, B_norm = normalize_along_axis(A), normalize_along_axis(B)
    D = torch.matmul(A_norm, A_norm.T)
    E = torch.matmul(B_norm, B_norm.T)

    identity = torch.eye(A.shape[0]).to(A.device)

    loss_C = F.mse_loss(A, B)
    loss_D = F.mse_loss(D, identity) * args.loss_scale_A
    loss_E = F.mse_loss(E, identity) * args.loss_scale_B

    return loss_C + loss_D + loss_E

class SigmaLoss_AE(nn.Module):
    def __init__(self, args):
        super(SigmaLoss_AE, self).__init__()
        self.args = args
        self.final_criteria = nn.CrossEntropyLoss()
        self.local_criteria = compute_SCL_loss_AE
        self.method = args.method

        self.max_pool = nn.MaxPool2d(2, 2)
        self.max_pool = nn.AvgPool2d(2, 2)
        self.embed_target = args.embed_target

    def forward(self, activations, signals, target, method="final"):
        if method == "local":
            loss = list()
            for idx, (act, sig) in enumerate(zip(activations[:-1], signals[:-1])):
                if len(act.shape) == 4 and len(sig.shape) == 2: sig = sig.view(sig.shape[0], sig.shape[1], act.shape[2], act.shape[3]) 
                if len(act.shape) == 2 and len(sig.shape) == 4: act = act.view(act.shape[0], act.shape[1], sig.shape[2], sig.shape[3])
                loss += [self.local_criteria(self.args, act, sig, target, predictions=activations[-1])]
            return sum(loss), loss[-1].item()
        elif method == "final":
            loss = self.final_criteria(activations[-1], target)
            return loss, loss.item()