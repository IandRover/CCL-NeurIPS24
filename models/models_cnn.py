from .layers import C2Linear, C2Conv, C2ConvTranspose, C2View
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()

        self.enc1 = C2Conv(args, args.num_chn, 128, 3)  # output size = 128 x 26 x 26
        self.enc2 = C2Conv(args, 128, 128, 3)  # output size = 256 x 24 x 24
        self.enc3 = C2Conv(args, 128, 256, 3)  # output size = 512 x 22 x 22
        self.enc4 = C2Conv(args, 256, 256, 3)  # output size = 1024 x 20 x 20
        self.enc5 = C2Conv(args, 256, 512, 3)  # output size = 1024 x 18 x 18
        self.view = C2View((512,1,1), 512*1*1)

        if args.dataset == "MNIST" or args.dataset == "FashionMNIST" or args.dataset == "CIFAR10" or args.dataset == "MNIST_CNN" or args.dataset == "FashionMNIST_CNN" or args.dataset == "STL10_cls":
            self.num_classes = 10
        elif args.dataset == "CIFAR100":
            self.num_classes = 100
        self.output = C2Linear(args, 512*1*1, self.num_classes)

        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.output]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params

    def forward(self, x, detach_grad=False):
        a1 = self.enc1(x, detach_grad)
        a2 = self.enc2(a1, detach_grad)
        a3 = self.enc3(a2, detach_grad)
        a4 = self.enc4(a3, detach_grad)
        a5 = self.enc5(a4, detach_grad)
        a6 = self.view(a5, detach_grad)
        b = self.output(a6, detach_grad, act=False)
        return [x, a1, a2, a3, a4, a6, b]
    
    def reverse(self, target, detach_grad=True):
        if len(target.shape) == 1: 
            target = F.one_hot(target, num_classes=self.num_classes).float().to(target.device)
        c6 = self.output.reverse(target, detach_grad)
        c5 = self.view.reverse(c6, detach_grad)
        c4 = self.enc5.reverse(c5, detach_grad)
        c3 = self.enc4.reverse(c4, detach_grad)
        c2 = self.enc3.reverse(c3, detach_grad)
        c1 = self.enc2.reverse(c2, detach_grad)
        c0 = self.enc1.reverse(c1, detach_grad, act=False)
        return [c0, c1, c2, c3, c4, c6, target]
    

def get_activation_function(func_name):
    if func_name == "tanh": return F.tanh
    elif func_name == "elu": return F.elu
    elif func_name == "relu": return F.relu
    elif func_name == "leaky_relu": return F.leaky_relu
    elif func_name == "selu": return F.selu
    else: raise ValueError(f"Activation function {func_name} not implemented")


class C2Pool(nn.Module):
    def __init__(self, args, kernel_size, stride, padding):
        super(C2Pool, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, detach_grad=False):
        return self.max_pool(x)

    def reverse(self, x, detach_grad=False):
        return self.upsample(x)
    
class C2ConvStride1(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel, fw_bias=False, bw_bias=False):
        super(C2ConvStride1, self).__init__()

        # convolutional layers
        self.forward_layer = nn.Conv2d(in_channels, out_channels, kernel, stride=1, padding=kernel//2, bias=fw_bias)
        self.backward_layer = nn.Conv2d(out_channels, in_channels, kernel, stride=1, padding=kernel//2, bias=bw_bias)
        
        # batchnorm
        self.fw_bn = args.fw_bn
        self.bw_bn = args.bw_bn
        if self.fw_bn == 1: self.forward_bn = nn.BatchNorm2d(in_channels)
        elif self.fw_bn == 2: self.forward_bn = nn.BatchNorm2d(out_channels)
        if self.bw_bn == 1: self.backward_bn = nn.BatchNorm2d(out_channels)
        elif self.bw_bn == 2: self.backward_bn = nn.BatchNorm2d(in_channels)

        # activation functions
        self.forward_act = get_activation_function(args.act_F)
        self.backward_act = get_activation_function(args.act_B)

        # weight initialization
        if args.bias_init == "zero" and self.forward_layer.bias is not None: self.forward_layer.bias.data.zero_()
        if args.bias_init == "zero" and self.backward_layer.bias is not None: self.backward_layer.bias.data.zero_()

    def get_parameters(self):
        self.forward_params = list(self.forward_layer.parameters())
        self.backward_params = list(self.backward_layer.parameters())
        return self.forward_params, self.backward_params
            
    def forward(self, x, detach_grad=False, act=True):

        # gradient detachment
        if detach_grad: x = x.detach()

        if not act: return self.forward_layer(x)

        # forward pass
        if self.fw_bn == 0: x = self.forward_layer(x)
        elif self.fw_bn == 1: x = self.forward_layer(self.forward_bn(x))
        elif self.fw_bn == 2: x = self.forward_bn(self.forward_layer(x))
        x = self.forward_act(x)

        return x

    def reverse(self, x, detach_grad=False, act=True):

        # gradient detachment
        if detach_grad: x = x.detach()

        if not act: return self.backward_layer(x)

        # backward pass
        if self.bw_bn == 0: x = self.backward_layer(x)
        elif self.bw_bn == 1: x = self.backward_layer(self.backward_bn(x))
        elif self.bw_bn == 2: x = self.backward_bn(self.backward_layer(x))
        x = self.backward_act(x)

        return x

class CNNModel_Pool(nn.Module):
    def __init__(self, args):
        super(CNNModel_Pool, self).__init__()

        self.enc1 = C2ConvStride1(args, args.num_chn, 128, 3)  # output size = 128 x 26 x 26
        self.pool1 = C2Pool(args, 2, 2, 0)  # output size = 128 x 13 x 13
        self.enc2 = C2ConvStride1(args, 128, 128, 3)  # output size = 128 x 13 x 13
        self.pool2 = C2Pool(args, 2, 2, 0)  # output size = 128 x 6 x 6
        self.enc3 = C2ConvStride1(args, 128, 256, 3)  # output size = 256 x 6 x 6
        self.pool3 = C2Pool(args, 2, 2, 0)  # output size = 256 x 3 x 3
        self.enc4 = C2ConvStride1(args, 256, 256, 3)  # output size = 256 x 3 x 3
        self.pool4 = C2Pool(args, 2, 2, 0)  # output size = 256 x 1 x 1
        self.enc5 = C2ConvStride1(args, 256, 512, 3)  # output size = 512 x 1 x 1
        self.pool5 = C2Pool(args, 2, 2, 0)  # output size = 512 x 1 x 1
        self.view = C2View((512,1,1), 512*1*1)
        

        if args.dataset == "MNIST" or args.dataset == "FashionMNIST" or args.dataset == "CIFAR10" or args.dataset == "STL10_cls":
            self.num_classes = 10
        elif args.dataset == "CIFAR100":
            self.num_classes = 100
        self.output = C2Linear(args, 512*1*1, self.num_classes)

        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.output]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params

    def forward(self, x, detach_grad=False):

        a1 = self.enc1(x, detach_grad)
        a1 = self.pool1(a1, detach_grad)
        a2 = self.enc2(a1, detach_grad)
        a2 = self.pool2(a2, detach_grad)
        a3 = self.enc3(a2, detach_grad)
        a3 = self.pool3(a3, detach_grad)
        a4 = self.enc4(a3, detach_grad)
        a4 = self.pool4(a4, detach_grad)
        a5 = self.enc5(a4, detach_grad)
        a5 = self.pool5(a5, detach_grad)
        a5 = self.view(a5, detach_grad)
        b = self.output(a5, detach_grad, act=False)
        return [x, a1, a2, a3, a4, a5, b]
        # return [x, a1, a2, a3, a4, a5, a6, b]
    
    def reverse(self, target, detach_grad=True):
        if len(target.shape) == 1: 
            target = F.one_hot(target, num_classes=self.num_classes).float().to(target.device)
        c6 = self.output.reverse(target, detach_grad)
        c5 = self.view.reverse(c6, detach_grad)
        c4 = self.pool5.reverse(c5, detach_grad)
        c4 = self.enc5.reverse(c4, detach_grad)
        c3 = self.pool4.reverse(c4, detach_grad)
        c3 = self.enc4.reverse(c3, detach_grad)
        c2 = self.pool3.reverse(c3, detach_grad)
        c2 = self.enc3.reverse(c2, detach_grad)
        c1 = self.pool2.reverse(c2, detach_grad)
        c1 = self.enc2.reverse(c1, detach_grad)
        c0 = self.pool1.reverse(c1, detach_grad)
        c0 = self.enc1.reverse(c0, detach_grad, act=False)
        return [c0, c1, c2, c3, c4, c6, target]
    

class CNNModelVis(nn.Module):
    def __init__(self, args):
        super(CNNModelVis, self).__init__()

        self.enc1 = C2ConvStride1(args, args.num_chn, 128, 5)  # output size = 128 x 26 x 26
        self.pool1 = C2Pool(args, 2, 2, 0)  # output size = 128 x 13 x 13
        self.enc2 = C2ConvStride1(args, 128, 128, 3)  # output size = 128 x 13 x 13
        self.pool2 = C2Pool(args, 2, 2, 0)  # output size = 128 x 6 x 6
        self.enc3 = C2ConvStride1(args, 128, 256, 3)  # output size = 256 x 6 x 6
        self.pool3 = C2Pool(args, 2, 2, 0)  # output size = 256 x 3 x 3
        self.enc4 = C2ConvStride1(args, 256, 256, 3)  # output size = 256 x 3 x 3
        self.pool4 = C2Pool(args, 2, 2, 0)  # output size = 256 x 1 x 1
        self.enc5 = C2ConvStride1(args, 256, 512, 3)  # output size = 512 x 1 x 1
        self.pool5 = C2Pool(args, 2, 2, 0)  # output size = 512 x 1 x 1
        self.view = C2View((512,1,1), 512*1*1)

        if args.dataset == "MNIST" or args.dataset == "FashionMNIST" or args.dataset == "CIFAR10" or args.dataset == "STL10_cls":
            self.num_classes = 10
        elif args.dataset == "CIFAR100":
            self.num_classes = 100
        self.output = C2Linear(args, 512*1*1, self.num_classes)

        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.output]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params

    def forward(self, x, detach_grad=False):

        a1 = self.enc1(x, detach_grad)
        a1 = self.pool1(a1, detach_grad)
        a2 = self.enc2(a1, detach_grad)
        a2 = self.pool2(a2, detach_grad)
        a3 = self.enc3(a2, detach_grad)
        a3 = self.pool3(a3, detach_grad)
        a4 = self.enc4(a3, detach_grad)
        a4 = self.pool4(a4, detach_grad)
        a5 = self.enc5(a4, detach_grad)
        a5 = self.pool5(a5, detach_grad)
        a5 = self.view(a5, detach_grad)
        b = self.output(a5, detach_grad, act=False)
        return [x, a1, a2, a3, a4, a5, b]
    
    def reverse(self, target, detach_grad=True):
        if len(target.shape) == 1: 
            target = F.one_hot(target, num_classes=self.num_classes).float().to(target.device)
        c6 = self.output.reverse(target, detach_grad)
        c5 = self.view.reverse(c6, detach_grad)
        c4 = self.pool5.reverse(c5, detach_grad)
        c4 = self.enc5.reverse(c4, detach_grad)
        c3 = self.pool4.reverse(c4, detach_grad)
        c3 = self.enc4.reverse(c3, detach_grad)
        c2 = self.pool3.reverse(c3, detach_grad)
        c2 = self.enc3.reverse(c2, detach_grad)
        c1 = self.pool2.reverse(c2, detach_grad)
        c1 = self.enc2.reverse(c1, detach_grad)
        c0 = self.pool1.reverse(c1, detach_grad)
        c0 = self.enc1.reverse(c0, detach_grad, act=False)
        return [c0, c1, c2, c3, c4, c6, target]