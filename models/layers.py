import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def get_activation_function(func_name):
    if func_name == "tanh": return F.tanh
    elif func_name == "elu": return F.elu
    elif func_name == "relu": return F.relu
    elif func_name == "leaky_relu": return F.leaky_relu
    elif func_name == "selu": return F.selu
    else: raise ValueError(f"Activation function {func_name} not implemented")

class C2Linear(nn.Module):
    def __init__(self, args, in_features, out_features, fw_bias=False, bw_bias=False):
        super(C2Linear, self).__init__()

        # linear layers
        self.forward_layer = nn.Linear(in_features, out_features, bias=fw_bias)
        self.backward_layer = nn.Linear(out_features, in_features, bias=bw_bias)
        self.forward_act = get_activation_function(args.act_F)
        self.backward_act = get_activation_function(args.act_B)
        
        # batchnorm
        self.fw_bn = args.fw_bn
        self.bw_bn = args.bw_bn
        if self.fw_bn == 1: self.forward_bn = nn.BatchNorm1d(in_features)
        elif self.fw_bn == 2: self.forward_bn = nn.BatchNorm1d(out_features)
        if self.bw_bn == 1: self.backward_bn = nn.BatchNorm1d(out_features)
        elif self.bw_bn == 2: self.backward_bn = nn.BatchNorm1d(in_features)

        # weight initialization
        if args.bias_init == "zero" and self.forward_layer.bias is not None: self.forward_layer.bias.data.zero_()
        if args.bias_init == "zero" and self.backward_layer.bias is not None: self.backward_layer.bias.data.zero_()

        # orthogonal initialization
        # if args.init == "orthogonal":
        # init.orthogonal_(self.forward_layer.weight)
        # init.orthogonal_(self.backward_layer.weight)

    def get_parameters(self):
        self.forward_params = list(self.forward_layer.parameters())
        self.backward_params = list(self.backward_layer.parameters())
        return self.forward_params, self.backward_params
    
    def forward(self, x, detach_grad=False, act=True):
        
        # gradient detachment
        if detach_grad: x = x.detach()

        if not act: return self.forward_layer(x)
        
        # forward pass
        if   self.fw_bn == 0: x = self.forward_layer(x)
        elif self.fw_bn == 1: x = self.forward_layer(self.forward_bn(x))
        elif self.fw_bn == 2: x = self.forward_bn(self.forward_layer(x))
        x = self.forward_act(x)
        
        return x
    
    def reverse(self, x, detach_grad=False, act=True):

        # gradient detachment
        if detach_grad: x = x.detach()

        if not act: return self.backward_layer(x)
        
        # backward pass
        if  self.bw_bn == 0: x = self.backward_layer(x)
        elif self.bw_bn == 1: x = self.backward_layer(self.backward_bn(x))
        elif self.bw_bn == 2: x = self.backward_bn(self.backward_layer(x))
        x = self.backward_act(x)
        
        return x

class C2Conv(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel, fw_bias=False, bw_bias=False):
        super(C2Conv, self).__init__()

        # convolutional layers
        self.forward_layer = nn.Conv2d(in_channels, out_channels, kernel, stride=2, padding=kernel//2, bias=True)
        self.backward_layer = nn.Conv2d(out_channels, in_channels, kernel, stride=1, padding=kernel//2, bias=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # batchnorm
        self.fw_bn = args.fw_bn
        self.bw_bn = args.bw_bn
        if self.fw_bn == 1: 
            self.forward_bn = nn.BatchNorm2d(in_channels, affine=args.bn_affine)
        elif self.fw_bn == 2: 
            self.forward_bn = nn.BatchNorm2d(out_channels, affine=args.bn_affine)
        else:
            print("Not using bn in conv forward")
        if self.bw_bn == 1: 
            self.backward_bn = nn.BatchNorm2d(out_channels, affine=args.bn_affine)
        elif self.bw_bn == 2: 
            self.backward_bn = nn.BatchNorm2d(in_channels, affine=args.bn_affine)
        else:
            print("Not using bn in conv backward")

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

        # upsample
        x = self.upsample(x)

        if not act: return self.backward_layer(x)

        # backward pass
        if self.bw_bn == 0: x = self.backward_layer(x)
        elif self.bw_bn == 1: x = self.backward_layer(self.backward_bn(x))
        elif self.bw_bn == 2: x = self.backward_bn(self.backward_layer(x))
        x = self.backward_act(x)

        return x
    
class C2ConvTranspose(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel, fw_bias=False, bw_bias=False):
        super(C2ConvTranspose, self).__init__()

        # convolutional layers
        self.forward_layer = nn.Conv2d(in_channels, out_channels, kernel, stride=1, padding=kernel//2, bias=fw_bias)
        self.backward_layer = nn.Conv2d(out_channels, in_channels, kernel, stride=2, padding=kernel//2, bias=bw_bias)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # batchnorm
        self.fw_bn = args.fw_bn
        self.bw_bn = args.bw_bn
        if self.fw_bn == 1: self.forward_bn = nn.BatchNorm2d(in_channels, affine=args.bn_affine)
        elif self.fw_bn == 2: self.forward_bn = nn.BatchNorm2d(out_channels, affine=args.bn_affine)
        if self.bw_bn == 1: self.backward_bn = nn.BatchNorm2d(out_channels, affine=args.bn_affine)
        elif self.bw_bn == 2: self.backward_bn = nn.BatchNorm2d(in_channels, affine=args.bn_affine)

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

        # upsample
        x = self.upsample(x)

        if not act: return self.forward_layer(x)

        # backward pass
        if self.fw_bn == 0: x = self.forward_layer(x)
        elif self.fw_bn == 1: x = self.forward_layer(self.forward_bn(x))
        elif self.fw_bn == 2: x = self.forward_bn(self.forward_layer(x))
        x = self.forward_act(x)

        return x

    def reverse(self, x, detach_grad=False, act=True):

        # gradient detachment
        if detach_grad: x = x.detach()

        if not act: return self.backward_layer(x)

        # forward pass
        if self.bw_bn == 0: x = self.backward_layer(x)
        elif self.bw_bn == 1: x = self.backward_layer(self.backward_bn(x))
        elif self.bw_bn == 2: x = self.backward_bn(self.backward_layer(x))
        x = self.backward_act(x)

        return x
        
class C2View(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(C2View, self).__init__()
        self.input_shape = input_shape if not isinstance(input_shape,int) else [input_shape]
        self.output_shape = output_shape if not isinstance(output_shape,int) else [output_shape]
    
    def forward(self, x, detach_grad=False):
        return x.view([len(x)] + list(self.output_shape))
    
    def reverse(self, x, detach_grad=False):
        return x.view([len(x)] + list(self.input_shape))
    
