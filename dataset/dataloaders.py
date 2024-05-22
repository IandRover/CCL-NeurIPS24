import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

def get_dataset(args):

    if args.dataset == "MNIST":
        # transform = transforms.Compose([transforms.ToTensor()])
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
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
        train_dataset_raw = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_dataset_raw, [0.9, 0.1])
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize * 8, shuffle=False)
        return train_loader, val_loader, test_loader
    
    if args.dataset == "FashionMNIST_CNN":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(32),
        ])
        train_dataset_raw = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_dataset_raw, [0.9, 0.1])
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
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
        train_dataset_raw = datasets.STL10(root='./data', split='train', download=True, transform=None)
        train_dataset_raw, val_dataset_raw = random_split(train_dataset_raw, [0.9, 0.1])

        # Apply transformations to the training dataset
        train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=train_transform)
        train_dataset.data = train_dataset_raw.dataset.data[train_dataset_raw.indices]
        train_dataset.labels = train_dataset_raw.dataset.labels[train_dataset_raw.indices]

        # Apply transformations to the validation dataset
        val_dataset = datasets.STL10(root='./data', split='train', download=True, transform=test_transform)
        val_dataset.data = val_dataset_raw.dataset.data[val_dataset_raw.indices]
        val_dataset.labels = val_dataset_raw.dataset.labels[val_dataset_raw.indices]

        test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

        return train_loader, val_loader, test_loader

    if args.dataset == "STL10_cls":
        
        print(f"Using Randomcrop, horizontalflip")

        # Define the transformations
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        # Load the STL-10 dataset without any transformations
        train_dataset_raw = datasets.STL10(root='./data', split='train', download=True, transform=None)
        train_dataset_raw, val_dataset_raw = random_split(train_dataset_raw, [0.9, 0.1])

        # Apply transformations to the training dataset
        train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=train_transform)
        train_dataset.data = train_dataset_raw.dataset.data[train_dataset_raw.indices]
        train_dataset.labels = train_dataset_raw.dataset.labels[train_dataset_raw.indices]

        # Apply transformations to the validation dataset
        val_dataset = datasets.STL10(root='./data', split='train', download=True, transform=test_transform)
        val_dataset.data = val_dataset_raw.dataset.data[val_dataset_raw.indices]
        val_dataset.labels = val_dataset_raw.dataset.labels[val_dataset_raw.indices]

        test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=test_transform)

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
    

    if args.dataset == 'CIFAR100':
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
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        return train_loader, val_loader, test_loader

    if args.dataset == "scene_parse_150":
        
        # from datasets import load_dataset
        # import torch
        # from PIL import Image
        # from torchvision.transforms import ToTensor
        # import os
        # from tqdm.auto import tqdm
        # from torchvision import transforms

        # def save_dataset(dataset, directory, num_examples, img_shape, ann_shape):
        #     # Ensure the directory exists
        #     os.makedirs(directory, exist_ok=True)
            
        #     # Initialize large tensors to hold all images and annotations
        #     # Assuming images and annotations have the same dimensions for simplicity
        #     images_tensor = torch.zeros((num_examples, *img_shape))
        #     annotations_tensor = torch.zeros((num_examples, *ann_shape))
            
        #     transformation = transforms.Compose([
        #                     transforms.CenterCrop(256),
        #                     transforms.ToTensor()
        #                 ])
            
        #     # Iterate through the dataset
        #     for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        #         image_tensor = transformation(example['image'])
        #         annotation_tensor = transformation(example['annotation'])
                
        #         # Place tensors in the preallocated large tensors
        #         images_tensor[i] = image_tensor
        #         annotations_tensor[i] = annotation_tensor

        #     # Save the large tensors to files
        #     torch.save(images_tensor, os.path.join("./data/scene_parse_data", f"{directory}_images.pt"))
        #     torch.save(annotations_tensor, os.path.join("./data/scene_parse_data", f"{directory}_annotations.pt"))

        # # Load the datasets
        # from datasets import load_dataset
        # test_dataset = load_dataset("scene_parse_150", split="validation[:512]", name="scene_parsing", trust_remote_code=True)
        # val_dataset = load_dataset("scene_parse_150", split="train[-512:]", name="scene_parsing", trust_remote_code=True)
        # train_dataset = load_dataset("scene_parse_150", split="train[:10000]", name="scene_parsing", trust_remote_code=True)

        # # Example usage, modify img_shape and ann_shape based on actual dimensions
        # save_dataset(test_dataset, 'test_data', 512, (3, 256, 256), (3, 256, 256))
        # save_dataset(val_dataset, 'val_data', 512, (3, 256, 256), (3, 256, 256))
        # save_dataset(train_dataset, 'train_data', 10000, (3, 256, 256), (3, 256, 256))

        from torch.utils.data import TensorDataset, DataLoader
        train_annotations = torch.load("./data/scene_parse_data/train_data_annotations.pt")
        train_images = torch.load("./data/scene_parse_data/train_data_images.pt")
        train_dataset = TensorDataset(train_images, train_annotations)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

        val_annotations = torch.load("./data/scene_parse_data/val_data_annotations.pt")
        val_images = torch.load("./data/scene_parse_data/val_data_images.pt")
        val_dataset = TensorDataset(val_images, val_annotations)
        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

        test_annotations = torch.load("./data/scene_parse_data/test_data_annotations.pt")
        test_images  = torch.load("./data/scene_parse_data/test_data_images.pt")
        test_dataset = TensorDataset(test_images, test_annotations)
        test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
        
        return train_loader, val_loader, test_loader
