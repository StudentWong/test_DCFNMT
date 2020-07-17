import torch.utils.data as data
import torch
from os.path import join, isdir, isfile
from train.config import TrackerConfig
import cv2
import json
import glob
import numpy as np


def gaussian_shaped_labels_bias(sigma, sz, bias):
    if bias[0] >= 0:
        xl = np.arange(sz[0] + 0 - bias[0], sz[0] + 0) - np.floor(float(sz[0]) / 2)
        xr = np.arange(0, sz[0] - bias[0] + 0) - np.floor(float(sz[0]) / 2)
    else:
        xl = np.arange(0 - bias[0], sz[0] + 0) - np.floor(float(sz[0]) / 2)
        xr = np.arange(0, 0 - bias[0]) - np.floor(float(sz[0]) / 2)
    x_bias = np.append(xl, xr)
    if bias[1] >= 0:
        yl = np.arange(sz[1] + 0 - bias[1], sz[1] + 0) - np.floor(float(sz[1]) / 2)
        yr = np.arange(0, sz[1] - bias[1] + 0) - np.floor(float(sz[1]) / 2)

    else:
        yl = np.arange(0 - bias[1], sz[1] + 0) - np.floor(float(sz[1]) / 2)
        yr = np.arange(0, 0 - bias[1]) - np.floor(float(sz[1]) / 2)
    y_bias = np.append(yl, yr)

    x, y = np.meshgrid(x_bias,
                       y_bias)
    d = x ** 2 + y ** 2

    g = np.exp(-0.5 / (sigma ** 2) * d)

    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 0), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 0), axis=1)
    return g.astype(np.float32)


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(0, sz[0] + 0) - np.floor(float(sz[0]) / 2),
                       np.arange(0, sz[1] + 0) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 0), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 0), axis=1)
    return g.astype(np.float32)


def assign_train_test(config, train_test_factor=0.2, temp_distru=0.1, search_distru=0.7):
    snp_len = config.data_use
    test_len = int(snp_len*train_test_factor)
    test_ind = np.random.choice(snp_len, test_len, replace=False).tolist()
    test_set = set(test_ind)
    all_set = set(np.arange(snp_len).tolist())
    train_set = all_set - test_set
    train_ind = list(train_set)

    train_data = VID(train_ind, config,
                     temp_distru=temp_distru, search_distru=search_distru)

    test_data = VID(test_ind, config,
                    temp_distru=temp_distru, search_distru=search_distru)
    return [train_data, train_ind, test_data, test_ind]



class VID(data.Dataset):
    def __init__(self, snp_index, config,
                 temp_distru=0.1, search_distru=0.7,
                 n=2, train=True):
        self.root = config.data_root
        self.snp_index = snp_index
        self.mean = np.expand_dims(np.expand_dims(np.array([127, 127, 117]), axis=1), axis=1).astype(np.float32)
        self.T_len = config.T
        self.temp_distru = temp_distru
        self.search_distru = search_distru
        self.n = n
        self.img_size = config.img_input_size
        self.res_size = [config.w_CNN_out, config.h_CNN_out]
        self.train = train
        self.output_sigma = config.output_sigma

    def __getitem__(self, item):
        x = np.zeros(shape=(self.T_len, 3, self.img_size[0], self.img_size[1]), dtype=np.float)
        z = np.zeros(shape=(self.T_len, 3, self.img_size[0], self.img_size[1]), dtype=np.float)

        x_bias = np.zeros(shape=(self.T_len, 2), dtype=np.int)
        z_bias = np.zeros(shape=(self.T_len, 2), dtype=np.int)

        if self.train:
            res = np.zeros(shape=(self.T_len, self.res_size[0], self.res_size[1]), dtype=np.float)

        index = self.snp_index[item % self.n]
        folder_name = "{:08d}".format(index)
        folder_frame_files = glob.glob(join(self.root, folder_name, "tp_f*"))
        folder_frame_num = len(folder_frame_files)

        begin_num = np.random.randint(0, folder_frame_num - self.T_len + 1)
        # print(begin_num)
        temp_name1 = 'tp_f{:05d}.jpg'.format(begin_num)
        temp_path = join(self.root, folder_name, temp_name1)
        x_bias[0] = [0, 0]
        imt = cv2.imread(temp_path)
        # cv2.imshow("1", imt)
        # cv2.waitKey(0)
        x[0] = np.transpose(imt, (2, 0, 1)).astype(np.float32) - self.mean
        for i in range(begin_num + 1, begin_num + self.T_len):
            rand_num = np.random.randint(0, self.n)
            name = 'f{:05d}_d{:1.1f}_n{:01d}*'.format(i,
                                                      self.temp_distru,
                                                      rand_num)
            temp_pathN = glob.glob(join(self.root, folder_name, name))
            xy_bias = temp_pathN[0].replace(self.root+'/'+folder_name+'/', '')
            xy_bias = xy_bias.replace(
                'f{:05d}_d{:1.1f}_n{:01d}_'.format(i,
                                                   self.temp_distru,
                                                   rand_num),
                ''
            )
            xy_bias = xy_bias.replace('.jpg', '')
            xy_bias = xy_bias.replace('x', '')
            xy_bias = xy_bias.replace('y', '')
            x_bias_str = int(xy_bias.split('_')[0])
            y_bias_str = int(xy_bias.split('_')[1])

            x_bias[i-begin_num] = [x_bias_str, y_bias_str]

            im = cv2.imread(temp_pathN[0])
            x[i-begin_num] = np.transpose(im, (2, 0, 1)).astype(np.float32) - self.mean
            # cv2.imshow("1", im)
            # cv2.waitKey(0)
            # x

        for i in range(begin_num, begin_num + self.T_len):
            rand_num = np.random.randint(0, self.n)
            name = 'f{:05d}_d{:1.1f}_n{:01d}*'.format(i,
                                                      self.search_distru,
                                                      rand_num)
            temp_pathN = glob.glob(join(self.root, folder_name, name))
            xy_bias = temp_pathN[0].replace(self.root+'/'+folder_name+'/', '')
            xy_bias = xy_bias.replace(
                'f{:05d}_d{:1.1f}_n{:01d}_'.format(i,
                                                   self.search_distru,
                                                   rand_num),
                ''
            )
            xy_bias = xy_bias.replace('.jpg', '')
            xy_bias = xy_bias.replace('x', '')
            xy_bias = xy_bias.replace('y', '')
            x_bias_str = int(xy_bias.split('_')[0])
            y_bias_str = int(xy_bias.split('_')[1])

            z_bias[i-begin_num] = [x_bias_str, y_bias_str]

            im = cv2.imread(temp_pathN[0])
            z[i-begin_num] = np.transpose(im, (2, 0, 1)).astype(np.float32) - self.mean

            if self.train:
                res[i-begin_num] = gaussian_shaped_labels_bias(self.output_sigma,
                                                                  self.res_size,
                                                                  z_bias[i-begin_num])

        if self.train:
            return x, z, res
        else:
            return x, z

    def __len__(self):
        return len(self.snp_index) * self.n


if __name__ == '__main__':
    config = TrackerConfig()
    train, train_i, test, test_i = assign_train_test(config)
    xt, zt, r = train[1]
    #xte, zte, r = test[1]
    print(r.shape)
    print(test_i)
    # data = VID([0, 3], config=config, train=False)
    # x, z = data[0]
    #
    for i in range(0, config.T):

        cv2.imshow("1", (xt[i]+train.mean).transpose((1, 2, 0)).astype(np.uint8))
        cv2.waitKey(0)
        cv2.imshow("1", (zt[i] + train.mean).transpose((1, 2, 0)).astype(np.uint8))
        cv2.waitKey(0)
        cv2.imshow("1", (r[i]))
        cv2.waitKey(0)
    #
    #     imxfft = torch.rfft(torch.tensor(x[0], dtype=torch.float).unsqueeze(0), signal_ndim=2, onesided=True)
    #
    #     imzfft = torch.rfft(torch.tensor(z[i], dtype=torch.float).unsqueeze(0), signal_ndim=2, onesided=True)
    #     #print(imxfft.shape)
    #
    #     def complex_mulconj(x, z):
    #         out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    #         out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    #         return torch.stack((out_real, out_imag), -1)
    #
    #
    #     def complex_mul(x, z):
    #         out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    #         out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    #         return torch.stack((out_real, out_imag), -1)
    #
    #     y = complex_mulconj(imxfft, imzfft)
    #     yy = torch.sum(complex_mulconj(imxfft, imzfft), dim=1, keepdim=True)
    #     response = torch.irfft(y, signal_ndim=2, onesided=True).numpy()
    #     res = (response - np.ones_like(response) * np.min(response)) / (np.max(response) - np.min(response))
    #
    #
    #     yc = torch.nn.functional.conv2d(torch.tensor(x[0], dtype=torch.float).unsqueeze(0),
    #                                     torch.tensor(z[i], dtype=torch.float).unsqueeze(0),
    #                                     stride=[1, 1], padding=[100, 100])[0].numpy()
    #     cv2.imshow("1", res[0].transpose(1, 2, 0))
    #     cv2.waitKey(0)
    #
    #     resc = (yc - np.ones_like(yc) * np.min(yc)) / (np.max(yc) - np.min(yc))
    #
    #     cv2.imshow("1", resc.transpose(1, 2, 0))
    #     cv2.waitKey(0)

