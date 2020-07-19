from torch import nn
from os.path import join
import numpy as np
import cv2
import torch
from torch.utils.checkpoint import checkpoint
import torch
from train.modules.NTM import NTM
from train.modules.feature import Feature
from config import TrackerConfig
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
        self.label_sum = config.label_sum.clone()
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

        norm_scal = torch.sum(response, dim=(2, 3), keepdim=True)
        response = (response/norm_scal.expand_as(response))*\
                   self.label_sum.to(device=zf_btcwh.device).expand_as(response)


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
        norm_scal = torch.sum(response, dim=(2, 3), keepdim=True)
        response = (response / norm_scal.expand_as(response)) * \
                   self.label_sum.to(device=zf_btcwh.device).expand_as(response)
        # print(response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out).is_contiguous())
        return response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out)

    def forward_batch(self, x_i, z_i):
        """
        :param x_i: batch * T * 3 * img_input_size1 * img_input_size2
        :param z_i: batch * T * 3 * img_input_size1 * img_input_size2
        :return: response batch * T * 1 * cnn_outsize1 * cnn_outsize2
        """
        if self.usecheckpoint:
            return self.forward_checkpoint(x_i, z_i)
        else:
            return self.forward_no_checkpoint(x_i, z_i)

    def forward(self, x_i, z_i, h_p=None, c_p=None):

        xf_btcwh = self.feature(x_i.unsqueeze(0))
        xf = xf_btcwh.permute(0, 2, 3, 1).contiguous()
        xf = xf.view((1, self.config.dim_C2_1, self.config.dim_C2_2))

        zf_btcwh = self.feature(z_i.unsqueeze(0))

        if h_p is None:
            h_p = torch.ones((1, self.config.dim_h_o), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_p = h_p.cuda()
        if c_p is None:
            c_p = xf

        h_o, c_o = self.ntm.forward(h_p, c_p, xf)

        c = c_o.permute(0, 2, 1).contiguous()
        # print(c.is_contiguous())
        c_btcwh = c.view((1, self.config.dim_C2_2,
                          self.config.w_CNN_out, self.config.h_CNN_out))

        cfft = torch.rfft(c_btcwh, signal_ndim=2)
        zfft = torch.rfft(zf_btcwh, signal_ndim=2)

        kzzf = torch.sum(torch.sum(cfft ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(zfft, cfft), dim=1, keepdim=True)
        alphaf = self.yf.to(device=z_i.device) / (kzzf + self.lambda0)  # very Ugly
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
        print(response.shape)
        # print(response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out).is_contiguous())
        return response.view(1, self.config.w_CNN_out, self.config.h_CNN_out), h_o, c_o

from torch.utils.data import DataLoader
from train.dataprepare.data import VID

criterion = nn.MSELoss(size_average=False).cuda()
if __name__ == "__main__":
    config = TrackerConfig()
    net = DCFNTM(config)
    ss = torch.load("/home/studentw/disk3/tracker/test_DCFNMT/train/work/model_best.pth.tar")
    # for param_tensor in ss["state_dict"]:
    #     print(param_tensor)
    net.load_state_dict(ss["state_dict"])
    net.train()
    net = net.cuda()
    data = VID([3, 7], temp_distru=0.1, config=config, train=True)

    train_loader = DataLoader(
    data, batch_size=config.batch * 1, shuffle=True,
    num_workers=1, pin_memory=True, drop_last=True)


    for i, (template, search, response) in enumerate(train_loader):
        # measure data loading time

        template = template.cuda(non_blocking=True).requires_grad_(True)
        search = search.cuda(non_blocking=True).requires_grad_(True)
        response = response.cuda(non_blocking=True).requires_grad_(True)
        # print(template.dtype)
        # compute output
        output = net.forward_batch(template, search)
        # print(output.shape)
        # print(response.shape)
        loss = criterion(output, response) / template.size(0)  # criterion = nn.MSEloss

        x = template.cpu().detach().numpy()[0]
        z = search.cpu().detach().numpy()[0]
        r = response.cpu().detach().numpy()[0]
        for i in range(0, 20):

            cv2.imshow("1", (x[i] + data.mean).transpose((1, 2, 0)).astype(np.uint8))
            cv2.waitKey(0)
            cv2.imshow("1", (z[i] + data.mean).transpose((1, 2, 0)).astype(np.uint8))
            cv2.waitKey(0)
            res = output.cpu().detach().numpy()[0][i]
            #resc = (res - np.ones_like(res) * np.min(res)) / (np.max(res) - np.min(res))
            print(res.shape)
            cv2.imshow("1", res)
            cv2.waitKey(0)
            cv2.imshow("1", (r[i]))
            cv2.waitKey(0)

    # for i in range(0, 20):
    #     if i == 0:
    #         res, h, c = net.forward(torch.tensor(x[i], dtype=torch.float),
    #                                 torch.tensor(z[i], dtype=torch.float))
    #     else:
    #         res, h, c = net.forward(torch.tensor(x[i], dtype=torch.float),
    #                                 torch.tensor(z[i], dtype=torch.float), h, c)
    #
    #     cv2.imshow("1", (x[i] + data.mean).transpose((1, 2, 0)).astype(np.uint8))
    #     cv2.waitKey(0)
    #     cv2.imshow("1", (z[i] + data.mean).transpose((1, 2, 0)).astype(np.uint8))
    #     cv2.waitKey(0)
    #     res = res.detach().numpy()[0]
    #     resc = (res - np.ones_like(res) * np.min(res)) / (np.max(res) - np.min(res))
    #     cv2.imshow("1", resc)
    #     cv2.waitKey(0)
    #     cv2.imshow("1", (r[i]))
    #     cv2.waitKey(0)





