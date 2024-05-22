from .layers import C2Linear, C2Conv, C2ConvTranspose, C2View
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel_MNIST(nn.Module):
    def __init__(self, args):
        super(LinearModel_MNIST, self).__init__()

        self.enc1 = C2Linear(args, 784, 256)  # output size = 256
        self.enc2 = C2Linear(args, 256, 256)  # output size = 256
        self.enc3 = C2Linear(args, 256, 256)  # output size = 256
        self.enc4 = C2Linear(args, 256, 256)  # output size = 256
        self.enc5 = C2Linear(args, 256, 256)  # output size = 256
        self.output = C2Linear(args, 256, 10)  # output size = 10

        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.output]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params

        # for enc_layer in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.output]:
        #     enc_layer.forward_layer.weight.data = torch.nn.init.orthogonal_(enc_layer.forward_layer.weight.data) 
        #     enc_layer.backward_layer.weight.data = torch.nn.init.orthogonal_(enc_layer.backward_layer.weight.data)
            
    def forward(self, x, detach_grad=False):
        a1 = self.enc1(x, detach_grad)
        a2 = self.enc2(a1, detach_grad)
        a3 = self.enc3(a2, detach_grad)
        a4 = self.enc4(a3, detach_grad)
        a5 = self.enc5(a4, detach_grad)
        b = self.output(a5, detach_grad, act=False)
        return [x, a1, a2, a3, a4, a5, b]
    
    def reverse(self, target, detach_grad=True):
        if len(target.shape) == 1: 
            target = F.one_hot(target, num_classes=10).float().to(target.device)
        c5 = self.output.reverse(target, detach_grad)
        c4 = self.enc5.reverse(c5, detach_grad)
        c3 = self.enc4.reverse(c4, detach_grad)
        c2 = self.enc3.reverse(c3, detach_grad)
        c1 = self.enc2.reverse(c2, detach_grad)
        c0 = self.enc1.reverse(c1, detach_grad, act=False)
        return [c0, c1, c2, c3, c4, c5, target]
    
# For CIFAR-10: 3 fully connected layers with 1024 hid- den units + 1 output softmax layer with 10 units.
class LinearModel_CIFAR10(nn.Module):
    def __init__(self, args):
        super(LinearModel_CIFAR10, self).__init__()

        self.enc1 = C2Linear(args, 3072, 1024)
        self.enc2 = C2Linear(args, 1024, 1024)
        self.enc3 = C2Linear(args, 1024, 1024)
        self.output = C2Linear(args, 1024, 10)

        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.output]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params

    def forward(self, x, detach_grad=False):
        a1 = self.enc1(x, detach_grad)
        a2 = self.enc2(a1, detach_grad)
        a3 = self.enc3(a2, detach_grad)
        b = self.output(a3, detach_grad, act=False)
        return [x, a1, a2, a3, b]
    
    def reverse(self, target, detach_grad=True):
        if len(target.shape) == 1: 
            target = F.one_hot(target, num_classes=10).float().to(target.device)
        c3 = self.output.reverse(target, detach_grad)
        c2 = self.enc3.reverse(c3, detach_grad)
        c1 = self.enc2.reverse(c2, detach_grad)
        c0 = self.enc1.reverse(c1, detach_grad, act=False)
        return [c0, c1, c2, c3, target]
    

class LinearModel_CIFAR100(nn.Module):
    def __init__(self, args):
        super(LinearModel_CIFAR100, self).__init__()

        self.enc1 = C2Linear(args, 3072, 1024)
        self.enc2 = C2Linear(args, 1024, 1024)
        self.enc3 = C2Linear(args, 1024, 1024)
        self.output = C2Linear(args, 1024, 100)

        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.output]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params
            
    def forward(self, x, detach_grad=False):
        a1 = self.enc1(x, detach_grad)
        a2 = self.enc2(a1, detach_grad)
        a3 = self.enc3(a2, detach_grad)
        b = self.output(a3, detach_grad, act=False)
        return [x, a1, a2, a3, b]
    
    def reverse(self, target, detach_grad=True):
        if len(target.shape) == 1: 
            target = F.one_hot(target, num_classes=100).float().to(target.device)
        c3 = self.output.reverse(target, detach_grad)
        c2 = self.enc3.reverse(c3, detach_grad)
        c1 = self.enc2.reverse(c2, detach_grad)
        c0 = self.enc1.reverse(c1, detach_grad, act=False)
        return [c0, c1, c2, c3, target]
    

class AEModel_L4(nn.Module):
    def __init__(self, args):
        super(AEModel_L4, self).__init__()

        self.enc1 = C2Conv(args, args.num_chn, 128, 3)  # output size = 64 x 14 x 14
        self.enc2 = C2Conv(args, 128, 256, 3)  # output size = 128 x 7 x 7
        self.enc3 = C2Conv(args, 256, 1024, 3)  # output size = 256 x 4 x 4
        self.enc4 = C2Conv(args, 1024, 2048, 3)  # output size = 256 x 2 x 2
        self.dec4 = C2ConvTranspose(args, 2048, 1024, 3)  # output size = 128 x 7 x 7
        self.dec3 = C2ConvTranspose(args, 1024, 256, 3)  # output size = 128 x 7 x 7
        self.dec2 = C2ConvTranspose(args, 256, 128, 3)  # output size = 64 x 14 x 14
        self.dec1 = C2ConvTranspose(args, 128, args.num_chn, 3)  # output size = 1 x 28 x 28
                
        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.enc4, self.dec4, self.dec3, self.dec2, self.dec1]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params

        for dec_layer in [self.dec4, self.dec3, self.dec2, self.dec1]:
            dec_layer.forward_layer.weight.data *= 0.001

        for enc_layer in [self.enc4, self.enc3, self.enc2, self.enc1]:
            enc_layer.backward_layer.weight.data *= 0.001

    def forward(self, x, detach_grad=False):

        a1 = self.enc1(x, detach_grad)
        a2 = self.enc2(a1, detach_grad)
        a3 = self.enc3(a2, detach_grad)
        a4 = self.enc4(a3, detach_grad)
        b4 = self.dec4(a4, detach_grad)
        b3 = self.dec3(b4, detach_grad)
        b2 = self.dec2(b3, detach_grad)
        b1 = self.dec1(b2, detach_grad, act=False)
        return [x, a1, a2, a3, a4, b4, b3, b2, b1]
    
    def reverse(self, target, detach_grad=True):
        if len(target.shape) == 1: 
            target = F.one_hot(target, num_classes=100).float().to(target.device)
        c2 = self.dec1.reverse(target, detach_grad)
        c3 = self.dec2.reverse(c2, detach_grad)
        c4 = self.dec3.reverse(c3, detach_grad)
        c5 = self.dec4.reverse(c4, detach_grad)
        d3 = self.enc4.reverse(c5, detach_grad)
        d2 = self.enc3.reverse(d3, detach_grad)
        d1 = self.enc2.reverse(d2, detach_grad)
        d0 = self.enc1.reverse(d1, detach_grad, act=False)
        return [d0, d1, d2, d3, c5, c4, c3, c2, target]