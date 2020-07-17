from torch import nn
import torch

def norm_grad(input, max_norm):
    if input.requires_grad:
        def norm_hook(grad):
            N = grad.size(0) # batch number
            norm = grad.contiguous().view(N, -1).norm(p=2, dim=1) + 1e-6
            scale = (norm / max_norm).clamp(min=1).view([N]+[1]*(grad.dim()-1))
            return grad / scale

            # clip_coef = float(max_norm) / (grad.norm(2).data[0] + 1e-6)
            # return grad.mul(clip_coef) if clip_coef < 1 else grad
        input.register_hook(norm_hook)


class IdentityFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return IdentityFun.apply(input)


def func(func_name):
    if func_name is None:
        return None
    elif func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    else:
        assert False, 'Invalid func_name.'


class FCN(nn.Module):

    def __init__(self, features, hid_trans='tanh', out_trans=None, hid_bn=0, out_bn=0):
        super(FCN, self).__init__()
        self.layer_num = len(features) - 1
        assert self.layer_num > 0, 'Invalid fc parameters'
        self.hid_bn = hid_bn
        self.out_bn = out_bn
        # Linear layers 快速构建多层FC
        for i in range(0, self.layer_num):
            setattr(self, 'fc'+str(i), nn.Linear(features[i], features[i+1]))
            if hid_bn == 1:
                setattr(self, 'hid_bn_func'+str(i), nn.BatchNorm1d(features[i+1]))
        if out_bn == 1:
            self.out_bn_func = nn.BatchNorm1d(features[-1])
        # Transformations
        self.hid_trans_func = func(hid_trans)
        self.out_trans_func = func(out_trans)

    def forward(self, X):
        H = X
        # Hidden layers
        for i in range(0, self.layer_num):
            H = getattr(self, 'fc'+str(i))(H)
            if i < self.layer_num - 1:
                if self.hid_bn == 1:
                    H = getattr(self, 'hid_bn_func'+str(i))(H)
                H = self.hid_trans_func(H)
        # Output layer
        if self.out_bn == 1:
            H = self.out_bn_func(H)
        if self.out_trans_func is not None:
            H = self.out_trans_func(H)
        return H