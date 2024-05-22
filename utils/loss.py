import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_centralization(model):
    with torch.no_grad():
        for p1, p2 in model.named_parameters():
            if "bias" in p1 or p2.grad is None: continue
            if len(p2.shape) == 2: p2.grad -= p2.grad.mean(dim=1,keepdim=True)
            elif len(p2.shape) == 4: p2.grad -= p2.grad.mean(dim=[1,2,3],keepdim=True) 

def normalize_along_axis(x, norm=True):
    x = x.reshape(len(x), -1)
    if norm:
        norm = torch.norm(x, dim=1, keepdim=True)
    return x / (norm + 1e-8)

def compute_scl_loss(args, A, B):

    loss1 = F.mse_loss(A, B) * A.shape[2] * args.loss_scale_C

    return loss1

class C2Loss_AE(nn.Module):
    def __init__(self, args):
        super(C2Loss_AE, self).__init__()
        self.args = args
        self.final_criteria = nn.CrossEntropyLoss()
        self.local_criteria = compute_scl_loss
        self.method = args.method

    def forward(self, activations, signals, target, method="final"):
        if method == "local":
            loss = list()
            for idx, (act, sig) in enumerate(zip(activations, signals)):
                if len(act.shape) == 4 and len(sig.shape) == 2: sig = sig.view(sig.shape[0], sig.shape[1], act.shape[2], act.shape[3]) 
                if len(act.shape) == 2 and len(sig.shape) == 4: act = act.view(act.shape[0], act.shape[1], sig.shape[2], sig.shape[3])
                loss += [self.local_criteria(self.args, act, sig)]
            return sum(loss), loss[-1].item()
        elif method == "final":
            loss = self.final_criteria(activations[-1], target)
            return loss, loss.item()
        
def compute_scl_loss_classification(args, A, B, target, predictions):

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
        
    A = A.view(A.shape[0], -1)
    B = B.view(B.shape[0], -1)
    C = A@B.T

    if len(B) == len(args.T10):
        identity = (target.unsqueeze(1) == args.T10).float().to(target.device)
    else:
        target_A = torch.zeros_like(C, dtype=torch.float32, device=target.device)
        target_A.scatter_(1, target.unsqueeze(1), 1.0)
        identity = torch.matmul(target_A, target_A.T)

    loss1 = F.mse_loss(C, identity) / A.shape[1] * args.loss_scale_C * 256

    return loss1
        
class C2Loss_Classification(nn.Module):
    def __init__(self, args):
        super(C2Loss_Classification, self).__init__()
        self.args = args
        self.final_criteria = nn.CrossEntropyLoss()
        self.local_criteria = compute_scl_loss_classification
        self.method = args.method

    def forward(self, activations, signals, target, method="final"):
        if method == "local":
            loss = list()
            for idx, (act, sig) in enumerate(zip(activations[:-1], signals[:-1])):
                if len(act.shape) == 4 and len(sig.shape) == 2: sig = sig.view(sig.shape[0], sig.shape[1], act.shape[2], act.shape[3]) 
                if len(act.shape) == 2 and len(sig.shape) == 4: act = act.view(act.shape[0], act.shape[1], sig.shape[2], sig.shape[3])
                loss += [self.local_criteria(self.args, act, sig, target, activations[-1])]
            loss += [self.final_criteria(activations[-1], target)]
            return sum(loss), loss[-1].item()
        elif method == "final":
            loss = self.final_criteria(activations[-1], target)
            return loss, loss.item()
        
        
def compute_scl_loss_classification_cnn_test(args, A, B, target, predictions):

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
        
    d = A.shape[1]
    A = normalize_along_axis(A)
    B = normalize_along_axis(B)
    C = A@B.T

    identity = (target.unsqueeze(1) == args.T10).float().to(target.device)
    loss1 = F.mse_loss(C, identity) * args.loss_scale_C

    identity = torch.eye(A.shape[0]).to(A.device)
    D = torch.matmul(A, A.T)
    loss2 = F.mse_loss(D, identity) * args.loss_scale_ssl

    identity = torch.eye(B.shape[0]).to(B.device)
    E = torch.matmul(B, B.T)
    loss3 = F.mse_loss(E, identity) * args.loss_scale_ssl

#     if torch.rand(1).item() < 0.01:
#         print(loss1.item(), loss2.item(), loss3.item(), )
        
    return loss1 + loss2 + loss3
        
class C2Loss_Classification_CNN_Test(nn.Module):
    def __init__(self, args):
        super(C2Loss_Classification_CNN_Test, self).__init__()
        self.args = args
        self.final_criteria = nn.CrossEntropyLoss()
        self.local_criteria = compute_scl_loss_classification_cnn_test
        self.method = args.method

    def forward(self, activations, signals, target, method="final"):
        if method == "local":
            loss = list()
            for idx, (act, sig) in enumerate(zip(activations[:-1], signals[:-1])):
                if len(act.shape) == 4 and len(sig.shape) == 2: sig = sig.view(sig.shape[0], sig.shape[1], act.shape[2], act.shape[3]) 
                if len(act.shape) == 2 and len(sig.shape) == 4: act = act.view(act.shape[0], act.shape[1], sig.shape[2], sig.shape[3])
                loss += [self.local_criteria(self.args, act, sig, target, activations[-1])]
            loss += [self.final_criteria(activations[-1], target)]
            return sum(loss), loss[-1].item()
        elif method == "final":
            loss = self.final_criteria(activations[-1], target)
            return loss, loss.item()
        





# Convolutional autoencoder
def compute_scl_loss_convautoencoder(args, A, B, target, predictions):
    # A_norm = normalize_along_axis(A)
    # B_norm = normalize_along_axis(B)
    # D = torch.matmul(A_norm, A_norm.T)
    # E = torch.matmul(B_norm, B_norm.T)

    # identity = torch.eye(A.shape[0]).to(A.device)
    # loss_C = F.mse_loss(A, B) * A.shape[2] * args.loss_scale_C / 2
    loss_C = F.mse_loss(A, B) * 1
    # loss_D = F.mse_loss(D, identity) * 0
    # loss_E = F.mse_loss(E, identity) * 0
    # loss_G = F.mse_loss(D, E) * args.loss_scale_B * A.shape[2] * 0.1

    return loss_C
    # loss = F.mse_loss(A, B) * A.size(2) / 10

    # C = normalize_along_axis(A)
    # D = normalize_along_axis(B)
    # loss += F.mse_loss(C, D)

    # return loss
        
class C2Loss_ConvAutoencoder(nn.Module):
    def __init__(self, args):
        super(C2Loss_ConvAutoencoder, self).__init__()
        self.args = args
        self.final_criteria = nn.CrossEntropyLoss()
        self.local_criteria = compute_scl_loss_convautoencoder
        self.method = args.method

    def forward(self, activations, signals, target):
        loss = list()
        for idx, (act, sig) in enumerate(zip(activations, signals)):
            if len(act.shape) == 4 and len(sig.shape) == 2: sig = sig.view(sig.shape[0], sig.shape[1], act.shape[2], act.shape[3]) 
            if len(act.shape) == 2 and len(sig.shape) == 4: act = act.view(act.shape[0], act.shape[1], sig.shape[2], sig.shape[3])
            loss += [self.local_criteria(self.args, act, sig, target, activations[-1])]
        return sum(loss), loss[-1].item()