import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation_function(func_name):
    if func_name == "tanh": return F.tanh
    elif func_name == "elu": return F.elu
    elif func_name == "relu": return F.relu
    elif func_name == "leaky_relu": return F.leaky_relu
    elif func_name == "selu": return F.selu
    else: raise ValueError(f"Activation function {func_name} not implemented")

class C2Conv(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel, stride, padding):
        super(C2Conv, self).__init__()
        self.forward_layer = nn.Conv2d(in_channels, out_channels, kernel, stride=1, padding=padding, bias=True)
        self.backward_layer = nn.Conv2d(out_channels, in_channels, kernel, stride=1, padding=kernel//2, bias=True)
        self.forward_bn = nn.BatchNorm2d(out_channels)
        self.backward_bn = nn.BatchNorm2d(in_channels)  
        self.forward_act = get_activation_function(args.act_F)
        self.backward_act = get_activation_function(args.act_B)

        self.pool = nn.MaxPool2d(2, 2)
        
        if self.forward_layer.bias is not None:
            self.forward_layer.bias.data.zero_()
    
    def get_parameters(self):
        self.forward_params = list(self.forward_layer.parameters())
        self.backward_params = list(self.backward_layer.parameters())
        return self.forward_params, self.backward_params
    
    def forward(self, x, detach_grad=False, act=True):
        if detach_grad: x = x.detach()
        if act == False:
            return self.forward_layer(x)
        return self.forward_act(self.forward_bn(self.pool(self.forward_layer(x))))
        
    def reverse(self, x, detach_grad=False, act=True):
        if detach_grad: x = x.detach()
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if act == False:
            return self.backward_layer(x)
        return self.backward_act(self.backward_bn(self.backward_layer(x)))
        
class C2ConvTrans(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel, stride, padding):
        super(C2ConvTrans, self).__init__()
        self.bn = 1
        self.forward_layer = nn.Conv2d(in_channels, out_channels, kernel, stride=1, padding=padding, bias=True)
        self.backward_layer = nn.Conv2d(out_channels, in_channels, kernel, stride=stride, padding=kernel//2, bias=True)
        if self.bn: self.forward_bn = nn.BatchNorm2d(out_channels)
        if self.bn: self.backward_bn1 = nn.BatchNorm2d(in_channels)
        self.forward_act = get_activation_function(args.act_F)
        self.backward_act = get_activation_function(args.act_B)
        
        if self.forward_layer.bias is not None:
            self.forward_layer.bias.data.zero_()
    
    def get_parameters(self):
        self.forward_params = list(self.forward_layer.parameters())
        self.backward_params = list(self.backward_layer.parameters())
        return self.forward_params, self.backward_params
    
    def forward(self, x, detach_grad=False, act=True):
        if detach_grad: x = x.detach()
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if act == False:
            return self.forward_layer(x)
        return self.forward_act(self.forward_bn(self.forward_layer(x)))
        
    def reverse(self, x, detach_grad=False, act=True):
        if detach_grad: x = x.detach()
        if act == False:
            return self.backward_layer(x)
        return self.backward_act(self.backward_bn1(self.backward_layer(x)))
    
class C2Model_ConvSeg_Legacy(nn.Module):
    def __init__(self, args):
        super(C2Model_ConvSeg_Legacy, self).__init__()

        self.enc1 =  C2Conv(args, args.num_chn, 256, 5, 2, 2)  # output size = 64 x 14 x 14
        self.enc2 =  C2Conv(args, 256, 256, 3, 2, 1)  # output size = 128 x 7 x 7        
        self.enc3 =  C2Conv(args, 256, 512, 3, 2, 1)  # output size = 256 x 4 x 4
        self.enc4 =  C2Conv(args, 512, 512, 3, 2, 1)  # output size = 256 x 2 x 2
        # self.enc5 =  C2Conv(args, 512, 512, 3, 2, 1)  # output size = 256 x 2 x 2
        # self.dec5 =  C2ConvTrans(args, 512, 512, 3, 2, 1)  # output size = 128 x 7 x 7
        self.dec4 =  C2ConvTrans(args, 512, 512, 3, 2, 1)  # output size = 128 x 7 x 7
        self.dec3 =  C2ConvTrans(args, 512, 256, 3, 2, 1)  # output size = 128 x 7 x 7
        self.dec2 =  C2ConvTrans(args, 256, 256, 3, 2, 1)  # output size = 64 x 14 x 14
        self.dec1 =  C2ConvTrans(args, 256, 251, 3, 2, 1)  # output size = 1 x 28 x 28

        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.enc4, 
                      self.dec4, self.dec3, self.dec2, self.dec1
                     ]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params
            
        for p1, p2 in self.named_parameters():
            if "weight" in p1 and "bn" not in p1:
                if p2.data.shape[0] >= p2.data.shape[1]:
                    p2.data = nn.init.orthogonal_(p2.data)
                else:
                    p2.data = nn.init.orthogonal_(p2.data.permute(1,0,2,3).contiguous()).permute(1,0,2,3).contiguous()

        for idx, dec_layer in enumerate([self.dec4, self.dec3, self.dec2, self.dec1]):
            dec_layer.forward_layer.weight.data *= 0.001
            dec_layer.backward_layer.weight.data *= 1
        
        for idx, enc_layer in enumerate([self.enc4, self.enc3, self.enc2, self.enc1]):
            enc_layer.backward_layer.weight.data *= 0.001
            enc_layer.forward_layer.weight.data *= 1
            
        self.dec1.forward_layer.weight.data *= 0
        self.enc1.backward_layer.weight.data *= 0

    def forward(self, x, detach_grad=False):           # 3 x 256 x 256
        a1 = self.enc1(x, detach_grad)   # 64 x 128 x 128
        a2 = self.enc2(a1, detach_grad)  # 128 x 64 x 64
        a3 = self.enc3(a2, detach_grad)  # 256 x 32 x 32
        a4 = self.enc4(a3, detach_grad)  # 256 x 16 x 16
        b4 = self.dec4(a4, detach_grad)  # 128 x 32 x 32
        b3 = self.dec3(b4+a3, detach_grad)  # 128 x 64 x 64
        b2 = self.dec2(b3+a2, detach_grad)
        b1 = self.dec1(b2, detach_grad, act=False)
        return [x, a1, a2, a3, a4, b4, b3, b2, b1]

    def reverse(self, target, detach_grad=True):
        
        target = F.one_hot((target[:,0]).long(), num_classes=251).permute(0,3,1,2).float().to(target.device)

        c2 = self.dec1.reverse(target, detach_grad) # 128 x 128 x 128
        c3 = self.dec2.reverse(c2, detach_grad)     # 128 x 64 x 64
        c4 = self.dec3.reverse(c3, detach_grad)
        c5 = self.dec4.reverse(c4, detach_grad)
        d3 = self.enc4.reverse(c5, detach_grad)
        d2 = self.enc3.reverse(d3, detach_grad)
        d1 = self.enc2.reverse(d2, detach_grad)
        d0 = self.enc1.reverse(d1, detach_grad, act=False)
        return [d0, d1, d2, d3, c5, c4, c3, c2, target]
    
class C2Model_ConvSeg_Legacy_v2(nn.Module):
    def __init__(self, args):
        super(C2Model_ConvSeg_Legacy_v2, self).__init__()

        self.enc1 =  C2Conv(args, args.num_chn, 256, 5, 2, 2)  # output size = 64 x 14 x 14
        self.enc2 =  C2Conv(args, 256, 256, 3, 2, 1)  # output size = 128 x 7 x 7        
        self.enc3 =  C2Conv(args, 256, 512, 3, 2, 1)  # output size = 256 x 4 x 4
        self.enc4 =  C2Conv(args, 512, 512, 3, 2, 1)  # output size = 256 x 2 x 2
        self.dec4 =  C2ConvTrans(args, 512, 512, 3, 2, 1)  # output size = 128 x 7 x 7
        self.dec3 =  C2ConvTrans(args, 512, 256, 3, 2, 1)  # output size = 128 x 7 x 7
        self.dec2 =  C2ConvTrans(args, 256, 256, 3, 2, 1)  # output size = 64 x 14 x 14
        self.dec1 =  C2ConvTrans(args, 256, 251+args.num_chn, 3, 2, 1)  # output size = 1 x 28 x 28

        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.enc1, self.enc2, self.enc3, self.enc4, 
                      self.dec4, self.dec3, self.dec2, self.dec1
                     ]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params
            
        for p1, p2 in self.named_parameters():
            if "weight" in p1 and "bn" not in p1:
                if p2.data.shape[0] >= p2.data.shape[1]:
                    p2.data = nn.init.orthogonal_(p2.data)
                else:
                    p2.data = nn.init.orthogonal_(p2.data.permute(1,0,2,3).contiguous()).permute(1,0,2,3).contiguous()

        for idx, dec_layer in enumerate([self.dec4, self.dec3, self.dec2, self.dec1]):
            dec_layer.forward_layer.weight.data *= 0.001
            dec_layer.backward_layer.weight.data *= 1
        
        for idx, enc_layer in enumerate([self.enc4, self.enc3, self.enc2, self.enc1]):
            enc_layer.backward_layer.weight.data *= 0.001
            enc_layer.forward_layer.weight.data *= 1
            
        self.dec1.forward_layer.weight.data *= 0
        self.enc1.backward_layer.weight.data *= 0

    def forward(self, x, detach_grad=False):           # 3 x 256 x 256
        a1 = self.enc1(x, detach_grad)   # 64 x 128 x 128
        a2 = self.enc2(a1, detach_grad)  # 128 x 64 x 64
        a3 = self.enc3(a2, detach_grad)  # 256 x 32 x 32
        a4 = self.enc4(a3, detach_grad)  # 256 x 16 x 16
        b4 = self.dec4(a4, detach_grad)  # 128 x 32 x 32
        b3 = self.dec3(b4+a3, detach_grad)  # 128 x 64 x 64
        b2 = self.dec2(b3+a2, detach_grad)
        b1 = self.dec1(b2, detach_grad, act=False)
        return [x, a1, a2, a3, a4, b4, b3, b2, b1]

    def reverse(self, target, detach_grad=True):
        
        target = F.one_hot((target[:,0]).long(), num_classes=251).permute(0,3,1,2).float().to(target.device)

        c2 = self.dec1.reverse(target, detach_grad) # 128 x 128 x 128
        c3 = self.dec2.reverse(c2, detach_grad)     # 128 x 64 x 64
        c4 = self.dec3.reverse(c3, detach_grad)
        c5 = self.dec4.reverse(c4, detach_grad)
        d3 = self.enc4.reverse(c5, detach_grad)
        d2 = self.enc3.reverse(d3, detach_grad)
        d1 = self.enc2.reverse(d2, detach_grad)
        d0 = self.enc1.reverse(d1, detach_grad, act=False)
        return [d0, d1, d2, d3, c5, c4, c3, c2, target]

def normalize_along_axis(x, y):
    x = x - torch.mean(x, dim=(2,3), keepdim=True)
    x = x / (torch.norm(x, dim=(2,3), keepdim=True) + 1e-8)
    y = y - torch.mean(y, dim=(2,3), keepdim=True)
    y = y / (torch.norm(y, dim=(2,3), keepdim=True) + 1e-8)
    return x, y 

def compute_SCL_loss_AE(args, A, B, layer=None):
    if layer == "end":
        loss_C = F.mse_loss(A, B) * A.shape[2] * args.loss_scale_C
        return loss_C
    else:
        loss_C = F.mse_loss(A, B) * A.shape[2] * args.loss_scale_C
        
        if args.loss_scale_D == 0: 
            loss_D = 0
        else:
            A_norm, B_norm = normalize_along_axis(A, B)
            loss_D = F.mse_loss(A_norm, B_norm) * A.shape[2] * args.loss_scale_D
    return loss_C + loss_D
        
class C2Loss_ConvSegmentation(nn.Module):
    def __init__(self, args):
        super(C2Loss_ConvSegmentation, self).__init__()
        self.args = args
        self.final_criteria = nn.CrossEntropyLoss()
        self.local_criteria = compute_SCL_loss_AE
        self.method = args.method

    def forward(self, activations, signals, target=None, method="final"):
        if method == "local":
            loss = list()
            for idx, (act, sig) in enumerate(zip(activations[1:-1], signals[1:-1])):
                if len(act.shape) == 4 and len(sig.shape) == 2: sig = sig.view(sig.shape[0], sig.shape[1], act.shape[2], act.shape[3]) 
                if len(act.shape) == 2 and len(sig.shape) == 4: act = act.view(act.shape[0], act.shape[1], sig.shape[2], sig.shape[3])
                loss += [self.local_criteria(self.args, act, sig)]
            loss += [self.local_criteria(self.args, activations[0], signals[0], layer="end")]
            loss += [self.final_criteria(activations[-1], target[:,0].long())]
            return sum(loss), loss[-1].item()
        elif method == "final":
            loss = self.final_criteria(activations[-1], target)
            return loss, loss.item()