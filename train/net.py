from torch import nn
from os.path import join
import numpy as np
import cv2
from torch.utils.checkpoint import checkpoint
import torch
from train.modules.NTM import NTM
from train.modules.feature import Feature
from train.config import TrackerConfig
from apex import amp
from pytorch_gpu_memory.gpu_memory_log import gpu_memory_log
from train.modules.common import FCN, norm_grad, Identity


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFNTM(nn.Module):

    def __init__(self, config, usecheckpoint=None):
        super(DCFNTM, self).__init__()
        self.config = config
        self.feature = Feature(config)
        self.lambda0 = config.lambda0
        self.yf = config.yf.clone()
        if usecheckpoint is None:
            self.usecheckpoint = True
        else:
            self.usecheckpoint = usecheckpoint
        self.ntm = NTM(config, self.usecheckpoint)

    def checkpoint_seg1_x(self, x_i):
        assert x_i.is_contiguous(), "view not contiguous"
        x = x_i.view(self.config.batch * self.config.T, 3, self.config.img_input_size[0], self.config.img_input_size[1])
        xf_btcwh = self.feature(x)
        return xf_btcwh

    def checkpoint_seg1_x_premute(self, xf_btcwh):
        xf = xf_btcwh.permute(0, 2, 3, 1).contiguous()
        assert xf.is_contiguous(), "view not contiguous"
        xf = xf.view((self.config.batch, self.config.T, self.config.dim_C2_1, self.config.dim_C2_2))
        return xf

    def checkpoint_seg1_z(self, z_i):
        assert z_i.is_contiguous(), "view not contiguous"
        z = z_i.view(self.config.batch * self.config.T, 3, self.config.img_input_size[0], self.config.img_input_size[1])
        zf_btcwh = self.feature(z)
        return zf_btcwh

    def checkpoint_seg_no_para(self, zf_btcwh, xf):
        if self.config.apex_level == "O2" or self.config.apex_level == "O3":
            h0 = torch.ones((self.config.batch, self.config.dim_h_o), dtype=torch.float).half()
        else:
            h0 = torch.ones((self.config.batch, self.config.dim_h_o), dtype=torch.float)

        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()
        h0 = h0.requires_grad_(True)
        c0 = xf[:, 0, :, :]

        h, c = self.ntm.forward_batch(h0, c0, xf)

        c = c.permute(0, 1, 3, 2).contiguous()
        # print(c.is_contiguous())
        assert c.is_contiguous(), "view not contiguous"
        c_btcwh = c.view((self.config.batch * self.config.T, self.config.dim_C2_2,
                          self.config.w_CNN_out, self.config.h_CNN_out))

        cfft = torch.rfft(c_btcwh, signal_ndim=2)
        zfft = torch.rfft(zf_btcwh, signal_ndim=2)

        kzzf = torch.sum(torch.sum(cfft ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(zfft, cfft), dim=1, keepdim=True)
        alphaf = self.yf.to(device=zf_btcwh.device) / (kzzf + self.lambda0)  # very Ugly
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
        # print(response.shape)
        # print(response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out).is_contiguous())
        assert response.is_contiguous(), "view not contiguous"
        return response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out)

    def forward_checkpoint(self, x_i, z_i):
        xf_btcwh = checkpoint(self.checkpoint_seg1_x, x_i)
        zf_btcwh = checkpoint(self.checkpoint_seg1_z, z_i)

        xf = checkpoint(self.checkpoint_seg1_x_premute, xf_btcwh)

        return self.checkpoint_seg_no_para(zf_btcwh, xf)

    def forward_no_checkpoint(self, x_i, z_i):
        xf_btcwh = self.checkpoint_seg1_x(x_i)
        zf_btcwh = self.checkpoint_seg1_z(z_i)

        xf = self.checkpoint_seg1_x_premute(xf_btcwh)

        return self.checkpoint_seg_no_para(zf_btcwh, xf)

    def forward_no_checkpoint_full_version(self, x_i, z_i):
        x = x_i.view(self.config.batch * self.config.T, 3, self.config.img_input_size[0], self.config.img_input_size[1])
        z = z_i.view(self.config.batch * self.config.T, 3, self.config.img_input_size[0], self.config.img_input_size[1])

        xf_btcwh = self.feature(x)
        xf = xf_btcwh.permute(0, 2, 3, 1).contiguous()
        xf = xf.view((self.config.batch, self.config.T, self.config.dim_C2_1, self.config.dim_C2_2))

        zf_btcwh = self.feature(z)

        h0 = torch.ones((self.config.batch, self.config.dim_h_o), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()
        c0 = xf[:, 0, :, :]

        h, c = self.ntm.forward_batch(h0, c0, xf)

        c = c.permute(0, 1, 3, 2).contiguous()
        # print(c.is_contiguous())
        c_btcwh = c.view((self.config.batch * self.config.T, self.config.dim_C2_2,
                          self.config.w_CNN_out, self.config.h_CNN_out))

        cfft = torch.rfft(c_btcwh, signal_ndim=2)
        zfft = torch.rfft(zf_btcwh, signal_ndim=2)

        kzzf = torch.sum(torch.sum(cfft ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(zfft, cfft), dim=1, keepdim=True)
        alphaf = self.yf.to(device=z.device) / (kzzf + self.lambda0)  # very Ugly
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
        # print(response.shape)
        # print(response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out).is_contiguous())
        return response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out)

    def forward(self, x_i, z_i):
        """
        :param x_i: batch * T * 3 * img_input_size1 * img_input_size2
        :param z_i: batch * T * 3 * img_input_size1 * img_input_size2
        :return: response batch * T * 1 * cnn_outsize1 * cnn_outsize2
        """
        if self.usecheckpoint:
            return self.forward_checkpoint(x_i, z_i)
        else:
            return self.forward_no_checkpoint(x_i, z_i)

import copy
# from graphviz import Digraph

import torch
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    config = TrackerConfig()
    net1 = DCFNTM(config, True)
    #net2 = DCFNTM(config, False).cuda()
    net2 = copy.deepcopy(net1)
    net2.usecheckpoint = False
    net2.ntm = copy.deepcopy(net1.ntm)
    net2.ntm.usecheckpoint = False

    net1 = net1.cuda()
    net2 = net2.cuda()

    x = torch.rand((config.batch, config.T, 3, config.img_input_size[0], config.img_input_size[1]), dtype=torch.float) * 1
    x = x.cuda()
    z = torch.rand((config.batch, config.T, 3, config.img_input_size[0], config.img_input_size[1]), dtype=torch.float) * 100
    z = z.cuda()


    # checkpoint
    x1 = x.clone().requires_grad_(True)
    z1 = z.clone().requires_grad_(True)
    losser1 = nn.MSELoss()
    optim1 = torch.optim.Adam(net1.parameters(), 1e-3)

    net1, optim1 = amp.initialize(net1, optim1, opt_level="O0")

    r1 = net1(x1, z1)
    rm1 = r1.mean(dim=(1, 2))
    loss1 = losser1(rm1, torch.ones_like(rm1, dtype=torch.float))

    optim1.zero_grad()

    with amp.scale_loss(loss1, optim1) as scaled_loss:
       scaled_loss.backward()

    #loss1.backward()
    optim1.step()
    for name, parms in net1.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
              # ' -->grad_value:', parms.grad,
              ' -->value:', parms.data)
    # gpu_memory_log()


    #no_checkpoint
    x2 = x.clone().requires_grad_(True)
    z2 = z.clone().requires_grad_(True)
    losser2 = nn.MSELoss()
    optim2 = torch.optim.Adam(net2.parameters(), 1e-3)
    r2 = net2.forward_no_checkpoint_full_version(x2, z2)
    rm2 = r2.mean(dim=(1,2))
    ##writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    ##writer.add_graph(net2, [x2, z2])
    ##writer.close()
    loss2 = losser2(rm2, torch.ones_like(rm2, dtype=torch.float))
    loss2.backward()
    optim2.step()
    optim2.zero_grad()
    for name, parms in net2.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
              #' -->grad_value:', parms.grad,
              ' -->value:', parms.data)
    # gpu_memory_log()


    # with torch.no_grad():
    #     r = net.forward(x, z)
    #     print(r)

    #gpu_memory_log()


