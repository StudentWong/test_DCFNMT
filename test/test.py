from torch import nn
from os.path import join
import numpy as np
import cv2
import torch
from torch.utils.checkpoint import checkpoint
import torch
from train.modules.NTM import NTM
from train.modules.feature import Feature

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


# class DCFNTM(nn.Module):
#
#     def __init__(self, config, usecheckpoint=None):
#         super(DCFNTM, self).__init__()
#         self.config = config
#         self.feature = Feature(config)
#         self.lambda0 = config.lambda0
#         # self.yf = config.yf.clone()
#         # self.label_sum = config.label_sum.clone()
#         if usecheckpoint is None:
#             self.usecheckpoint = True
#         else:
#             self.usecheckpoint = usecheckpoint
#
#         if config.C_norm:
#             self.init_x_norm = torch.nn.LayerNorm([config.dim_C2_1, config.dim_C2_2],
#                                                   elementwise_affine=config.norm_learnable)
#         self.ntm = NTM(config, self.usecheckpoint)
#
#     def checkpoint_seg1_x(self, x_i):
#         assert x_i.is_contiguous(), "view not contiguous"
#         x = x_i.view(self.config.batch * self.config.T, 3, self.config.img_input_size[0], self.config.img_input_size[1])
#         xf_btcwh = self.feature(x)
#         return xf_btcwh
#
#     def checkpoint_seg1_x_premute(self, xf_btcwh):
#         xf = xf_btcwh.permute(0, 2, 3, 1).contiguous()
#         assert xf.is_contiguous(), "view not contiguous"
#         xf = xf.view((self.config.batch, self.config.T, self.config.dim_C2_1, self.config.dim_C2_2))
#         return xf
#
#     def checkpoint_seg1_z(self, z_i):
#         assert z_i.is_contiguous(), "view not contiguous"
#         z = z_i.view(self.config.batch * self.config.T, 3, self.config.img_input_size[0], self.config.img_input_size[1])
#         zf_btcwh = self.feature(z)
#         return zf_btcwh
#
#     def checkpoint_seg_no_para(self, zf_btcwh, xf):
#         if self.config.apex_level == "O2" or self.config.apex_level == "O3":
#             h0 = torch.ones((self.config.batch, self.config.dim_h_o), dtype=torch.float).half()
#         else:
#             h0 = torch.ones((self.config.batch, self.config.dim_h_o), dtype=torch.float)
#
#         if next(self.parameters()).is_cuda:
#             h0 = h0.cuda()
#         h0 = h0.requires_grad_(True)
#         c0 = xf[:, 0, :, :]
#         c0 = self.init_x_norm(c0)
#         h, c = self.ntm.forward_batch(h0, c0, xf)
#
#         c = c.permute(0, 1, 3, 2).contiguous()
#         # print(c.is_contiguous())
#         assert c.is_contiguous(), "view not contiguous"
#         c_btcwh = c.view((self.config.batch * self.config.T, self.config.dim_C2_2,
#                           self.config.w_CNN_out, self.config.h_CNN_out))
#
#         cfft = torch.rfft(c_btcwh, signal_ndim=2)
#         zfft = torch.rfft(zf_btcwh, signal_ndim=2)
#
#         kzzf = torch.sum(torch.sum(cfft ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
#         kxzf = torch.sum(complex_mulconj(zfft, cfft), dim=1, keepdim=True)
#         alphaf = self.yf.clone().to(device=zf_btcwh.device) / (kzzf + self.lambda0)  # very Ugly
#         response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
#
#         # norm_scal = torch.sum(response, dim=(2, 3), keepdim=True)
#         # response = (response/norm_scal.expand_as(response))*\
#         #            self.label_sum.to(device=zf_btcwh.device).expand_as(response)
#
#         # print(response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out).is_contiguous())
#         assert response.is_contiguous(), "view not contiguous"
#         return response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out)
#
#     def forward_checkpoint(self, x_i, z_i):
#         xf_btcwh = checkpoint(self.checkpoint_seg1_x, x_i)
#         zf_btcwh = checkpoint(self.checkpoint_seg1_z, z_i)
#
#         xf = checkpoint(self.checkpoint_seg1_x_premute, xf_btcwh)
#
#         return self.checkpoint_seg_no_para(zf_btcwh, xf)
#
#     def forward_no_checkpoint(self, x_i, z_i):
#         xf_btcwh = self.checkpoint_seg1_x(x_i)
#         zf_btcwh = self.checkpoint_seg1_z(z_i)
#
#         xf = self.checkpoint_seg1_x_premute(xf_btcwh)
#
#         return self.checkpoint_seg_no_para(zf_btcwh, xf)
#
#     def forward_no_checkpoint_full_version(self, x_i, z_i):
#         x = x_i.view(self.config.batch * self.config.T, 3, self.config.img_input_size[0], self.config.img_input_size[1])
#         z = z_i.view(self.config.batch * self.config.T, 3, self.config.img_input_size[0], self.config.img_input_size[1])
#
#         xf_btcwh = self.feature(x)
#         xf = xf_btcwh.permute(0, 2, 3, 1).contiguous()
#         xf = xf.view((self.config.batch, self.config.T, self.config.dim_C2_1, self.config.dim_C2_2))
#
#         zf_btcwh = self.feature(z)
#
#         h0 = torch.ones((self.config.batch, self.config.dim_h_o), dtype=torch.float)
#         if next(self.parameters()).is_cuda:
#             h0 = h0.cuda()
#         c0 = xf[:, 0, :, :]
#
#         h, c = self.ntm.forward_batch(h0, c0, xf)
#
#         c = c.permute(0, 1, 3, 2).contiguous()
#         # print(c.is_contiguous())
#         c_btcwh = c.view((self.config.batch * self.config.T, self.config.dim_C2_2,
#                           self.config.w_CNN_out, self.config.h_CNN_out))
#
#         cfft = torch.rfft(c_btcwh, signal_ndim=2)
#         zfft = torch.rfft(zf_btcwh, signal_ndim=2)
#
#         kzzf = torch.sum(torch.sum(cfft ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
#         kxzf = torch.sum(complex_mulconj(zfft, cfft), dim=1, keepdim=True)
#         alphaf = self.yf.clone().to(device=z.device) / (kzzf + self.lambda0)  # very Ugly
#         response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
#         # norm_scal = torch.sum(response, dim=(2, 3), keepdim=True)
#         # response = (response / norm_scal.expand_as(response)) * \
#         #            self.label_sum.to(device=zf_btcwh.device).expand_as(response)
#         # print(response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out).is_contiguous())
#         return response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out)
#
#     def forward_batch(self, x_i, z_i):
#         """
#         :param x_i: batch * T * 3 * img_input_size1 * img_input_size2
#         :param z_i: batch * T * 3 * img_input_size1 * img_input_size2
#         :return: response batch * T * 1 * cnn_outsize1 * cnn_outsize2
#         """
#         if self.usecheckpoint:
#             return self.forward_checkpoint(x_i, z_i)
#         else:
#             return self.forward_no_checkpoint(x_i, z_i)
#
#     def forward(self, x_i, z_i, h_p=None, c_p=None):
#
#         xf_btcwh = self.feature(x_i.unsqueeze(0))
#         xf = xf_btcwh.permute(0, 2, 3, 1).contiguous()
#         xf = xf.view((1, self.config.dim_C2_1, self.config.dim_C2_2))
#
#         zf_btcwh = self.feature(z_i.unsqueeze(0))
#
#         if h_p is None:
#             h_p = torch.ones((1, self.config.dim_h_o), dtype=torch.float)
#         if next(self.parameters()).is_cuda:
#             h_p = h_p.cuda()
#         if c_p is None:
#             c_p = self.init_x_norm(xf)
#
#         h_o, c_o = self.ntm.forward(h_p, c_p, xf)
#
#         c = c_o.permute(0, 2, 1).contiguous()
#         # print(c.is_contiguous())
#         c_btcwh = c.view((1, self.config.dim_C2_2,
#                           self.config.w_CNN_out, self.config.h_CNN_out))
#
#         cfft = torch.rfft(c_btcwh, signal_ndim=2)
#         zfft = torch.rfft(zf_btcwh, signal_ndim=2)
#
#         if not self.config.direct_correlation:
#             kzzf = torch.sum(torch.sum(cfft ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
#             kxzf = torch.sum(complex_mulconj(zfft, cfft), dim=1, keepdim=True)
#             alphaf = self.yf.to(device=z_i.device) / (kzzf + self.lambda0)  # very Ugly
#             response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
#         else:
#             response_fft = torch.sum(complex_mulconj(zfft, cfft), dim=1, keepdim=True)
#             response = torch.irfft(response_fft, signal_ndim=2)
#         # norm_scal = torch.sum(response, dim=(2, 3), keepdim=True)
#         # response = (response / norm_scal.expand_as(response)) * \
#         #           self.label_sum.to(device=zf_btcwh.device).expand_as(response)
#         # print(response.shape)
#         # print(response.view(self.config.batch, self.config.T, self.config.w_CNN_out, self.config.h_CNN_out).is_contiguous())
#         return response.view(1, self.config.w_CNN_out, self.config.h_CNN_out), h_o, c_o


from torch.utils.data import DataLoader
from train.dataprepare.data import VID
import json
import time
from train.net import DCFNTM

criterion = nn.MSELoss(size_average=False).cuda()


def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0] - sz[0] / 2 + 1, pos[1] - sz[1] / 2 + 1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    return np.array([rect[0] + rect[2] / 2 - 1, rect[1] + rect[3] / 2 - 1]), np.array([rect[2], rect[3]])  # 0-index


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])  # 0-index


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


from runs.config1 import TrackerConfig
import glob

if __name__ == "__main__":

    base = '/home/studentw/disk3/OTB100/Skiing/'
    begin_num = 1

    use_gpu = True
    visualization = True

    # default parameter and load feature extractor network
    config = TrackerConfig()

    # exit()
    net = DCFNTM(config).cuda()
    ss = torch.load("/home/studentw/work1/model_best.pth.tar")
    # for param_tensor in ss["state_dict"]:
    #     print(param_tensor)
    net.load_state_dict(ss["state_dict"])
    net.eval()

    speed = []
    # loop videos

    folder_frame_files = glob.glob(join(base, 'img', '*.jpg'))
    folder_frame_num = len(folder_frame_files)

    # image_files = [join(base, 'img', 'img', im_f) for im_f in annos[video]['image_files']]

    tic = time.time()  # time start

    initpos = [446, 181, 29, 26]
    target_pos = np.array([initpos[0] + initpos[2] / 2, initpos[1] + initpos[3] / 2], dtype=np.float)
    target_sz = np.array([initpos[2], initpos[3]], dtype=np.float)
    # print(image_files[0])
    image_files = join(base, 'img', '{:04d}.jpg'.format(begin_num))
    im = cv2.imread(image_files)  # HxWxC

    # confine results
    # min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
    # max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

    # crop template
    window_sz = target_sz * (1 + config.padding)
    bbox = cxy_wh_2_bbox(target_pos, window_sz)
    patch = crop_chw(im, bbox, config.img_input_size[0])
    # cv2.imshow("1", patch.transpose((1, 2, 0)))
    # cv2.waitKey(0)
    target = patch - config.net_average_image
    temp = torch.tensor(target, dtype=torch.float).unsqueeze(0).unsqueeze(0).expand(1, config.T, 3,
                                                                                    config.img_input_size[0],
                                                                                    config.img_input_size[1])
    hp = None
    cp = None
    # patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
    for f in range(1 + begin_num, folder_frame_num + begin_num):  # track
        search_sz = target_sz * (1 + config.padding)
        search_pos = target_pos
        search_box = cxy_wh_2_bbox(search_pos, search_sz)

        image_files = join(base, 'img', '{:04d}.jpg'.format(f))
        im = cv2.imread(image_files)

        search_show = crop_chw(im, search_box, config.img_input_size[0])
        # cv2.imshow("1", search_show.transpose((1, 2, 0)))
        # cv2.waitKey(0)

        search_crop = crop_chw(im, search_box, config.img_input_size[0]) - config.net_average_image
        search = torch.tensor(search_crop, dtype=torch.float).unsqueeze(0).unsqueeze(0)

        # exit()

        with torch.no_grad():
            response, c, cmult = net.forward(x_i=temp.cuda(), z_i=search.cuda(), c_0=cp)
        # cview = c.permute(0, 2, 1).view((1, 32, 99, 99))
        # cview = cview[0, 0:3, :, :].permute(1, 2, 0).cpu().detach().numpy()
        # # cview = cview
        # cview = (cview - np.ones_like(cview) * np.min(cview)) / (np.max(cview) - np.min(cview))
        #
        # cb = cb[0, 0:3, :, :].permute(1, 2, 0).cpu().detach().numpy()
        # cb = (cb - np.ones_like(cb) * np.min(cb)) / (np.max(cb) - np.min(cb))
        r = response.permute(0, 2, 3, 1).cpu().detach().numpy()
        r = (r - np.ones_like(r) * np.min(r)) / (np.max(r) - np.min(r))
        # print(r.shape)
        cv2.imshow("r", r[0])
        cv2.waitKey(0)
        # exit()
        # cv2.imshow("1", cview)
        # cv2.waitKey(0)
        # cv2.imshow("1", cb)
        # cv2.waitKey(0)
        # for i in range(config.num_scale):  # crop multi-scale search region
        #     window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
        #     bbox = cxy_wh_2_bbox(target_pos, window_sz)
        #     patch_crop[i, :] = crop_chw(im, bbox, config.crop_sz)
        #
        #     search = patch_crop[i, :] - config.net_average_image
        #     response, h, c = net(torch.Tensor(search).cuda())

        peak, idx = torch.max(response.view(1, -1), 1)
        peak = peak.data.cpu().numpy()
        # best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx.cpu(), config.net_input_size)
        if r_max > config.net_input_size[0] / 2:
            r_max = r_max - config.net_input_size[0]
        #     if r_max[0] < -10:
        #         r_max[0] = -10
        # else:
        #     if r_max[0] > 10:
        #         r_max[0] = 10

        if c_max > config.net_input_size[1] / 2:
            c_max = c_max - config.net_input_size[1]
        #     if c_max[0] < -10:
        #         c_max[0] = -10
        # else:
        #     if c_max[0] > 10:
        #         c_max[0] = 10

        window_sz = target_sz * (1 + config.padding)
        # print(np.array([c_max[0], r_max[0]]))
        # print(target_pos)
        # print(np.array([c_max, r_max]) * window_sz) #/ config.img_input_size[0])
        target_pos = target_pos + np.array([c_max[0], r_max[0]]) * window_sz / config.img_input_size[0]
        # print(target_pos)
        target_sz = window_sz / (1 + config.padding)

        # model update
        temp_latest_box = cxy_wh_2_bbox(target_pos, window_sz)
        tempshow = crop_chw(im, temp_latest_box, config.img_input_size[0])
        # cv2.imshow("1", tempshow.transpose((1, 2, 0)))
        # cv2.waitKey(0)
        # exit()
        temp_latest_crop = crop_chw(im, temp_latest_box, config.img_input_size[0]) - config.net_average_image
        temp_latest = torch.tensor(temp_latest_crop, dtype=torch.float)

        temp_new = torch.zeros_like(temp, dtype=torch.float)
        temp_new[:, 0:config.T-1, :, :, :] = temp[:, 1:config.T, :, :, :]
        temp_new[0, config.T - 1, :, :, :] = temp_latest

        # res.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index
        cp = c
        if visualization:
            im_show = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im_show, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                          (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                          (0, 255, 0), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("1", im_show)
            cv2.waitKey(0)

    toc = time.time() - tic
    fps = folder_frame_num / toc
    speed.append(fps)
    print('{:3d} Video: {:12s} Time: {:3.1f}s\tSpeed: {:3.1f}fps'.format(video_id, video, toc, fps))

    # save result
    test_path = join('result', dataset, 'DCFNet_test')
    # if not isdir(test_path): makedirs(test_path)
    result_path = join(test_path, video + '.txt')
    with open(result_path, 'w') as f:
        for x in res:
            f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')

    print('***Total Mean Speed: {:3.1f} (FPS)***'.format(np.mean(speed)))

    # eval_auc(dataset, 'DCFNet_test', 0, 1)

    # config = TrackerConfig()
    # net = DCFNTM(config)
    # ss = torch.load("/home/studentw/disk3/tracker/test_DCFNMT/work/multimodule_normC_.tar")
    # # for param_tensor in ss["state_dict"]:
    # #     print(param_tensor)
    # net.load_state_dict(ss["state_dict"])
    # net.train()
    # net = net.cuda()
    # data = VID([1488],  config=config, train=True)
    #
    # train_loader = DataLoader(
    # data, batch_size=config.batch * 1, shuffle=True,
    # num_workers=1, pin_memory=True, drop_last=True)
    # (x, z, r) = data[0]
    #
    #
    # for i, (template, search, response) in enumerate(train_loader):
    #     # measure data loading time
    #
    #     template = template.cuda(non_blocking=True).requires_grad_(True)
    #     search = search.cuda(non_blocking=True).requires_grad_(True)
    #     response = response.cuda(non_blocking=True).requires_grad_(True)
    #     # print(template.dtype)
    #     # compute output
    #     output = net.forward(template, search)
    #     # print(output.shape)
    #     # print(response.shape)
    #     loss = criterion(output, response) / template.size(0)  # criterion = nn.MSEloss
    #
    # x = template.cpu().detach().numpy()[0]
    # z = search.cpu().detach().numpy()[0]
    # r = response.cpu().detach().numpy()[0]
    # for i in range(0, 12):
    #
    #     cv2.imshow("1", (x[i] + data.mean).transpose((1, 2, 0)).astype(np.uint8))
    #     cv2.waitKey(0)
    #     cv2.imshow("1", (z[i] + data.mean).transpose((1, 2, 0)).astype(np.uint8))
    #     cv2.waitKey(0)
    #     res = output.cpu().detach().numpy()[0][i]
    #     #resc = (res - np.ones_like(res) * np.min(res)) / (np.max(res) - np.min(res))
    #     print(res.shape)
    #     cv2.imshow("1", res)
    #     cv2.waitKey(0)
    #     cv2.imshow("1", (r[i]))
    #     cv2.waitKey(0)

    # for i in range(0, 11):
    #     with torch.no_grad():
    #         if i == 0:
    #             res, h, c = net.forward(torch.tensor(x[i], dtype=torch.float),
    #                                     torch.tensor(z[i], dtype=torch.float))
    #         else:
    #             res, h, c = net.forward(torch.tensor(x[i], dtype=torch.float),
    #                                     torch.tensor(z[i], dtype=torch.float), h, c)
    #
    #     cv2.imshow("1", (x[i] + data.mean).transpose((1, 2, 0)).astype(np.uint8))
    #     cv2.waitKey(0)
    #     cv2.imshow("1", (z[i] + data.mean).transpose((1, 2, 0)).astype(np.uint8))
    #     cv2.waitKey(0)
    #     res = res.detach().numpy()[0]
    #     #resc = (res - np.ones_like(res) * np.min(res)) / (np.max(res) - np.min(res))
    #     cv2.imshow("1", res)
    #     cv2.waitKey(0)
    #     cv2.imshow("1", (r[i]))
    #     cv2.waitKey(0)
