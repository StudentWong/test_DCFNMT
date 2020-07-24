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


def assign_train_test(config, train_test_factor=0.2, temp_distru=[0.1, 0.3, 0.5],
                      search_distru=[0.5, 0.7]):
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
                 temp_distru=[0.1, 0.3], search_distru=[0.5, 0.7],
                 n=2, train=True):
        self.root = config.data_root
        self.snp_index = snp_index
        self.mean = config.net_average_image
        self.T_len = config.T
        self.temp_distru = temp_distru
        self.search_distru = search_distru
        self.n = n
        self.img_size = config.img_input_size
        self.res_size = [config.w_CNN_out, config.h_CNN_out]
        self.train = train
        self.output_sigma = config.output_sigma

    def __getitem__(self, item):
        x = np.zeros(shape=(self.T_len, 3, self.img_size[0], self.img_size[1]), dtype=np.float32)
        z = np.zeros(shape=(self.T_len, 3, self.img_size[0], self.img_size[1]), dtype=np.float32)

        x_bias = np.zeros(shape=(self.T_len, 2), dtype=np.int)
        z_bias = np.zeros(shape=(self.T_len, 2), dtype=np.int)

        if self.train:
            res = np.zeros(shape=(self.T_len, self.res_size[0], self.res_size[1]), dtype=np.float32)

        index = self.snp_index[int(item / self.n)]

        folder_name = "{:08d}".format(index)
        folder_frame_files = glob.glob(join(self.root, folder_name, "tp_f*"))
        folder_frame_num = len(folder_frame_files)

        begin_num = np.random.randint(0, folder_frame_num - self.T_len + 1)

        stride = int((folder_frame_num-begin_num+1)/self.T_len)

        temp_name1 = 'tp_f{:05d}.jpg'.format(begin_num)
        temp_path = join(self.root, folder_name, temp_name1)
        x_bias[0] = [0, 0]
        imt = cv2.imread(temp_path)
        # cv2.imshow("1", imt)
        # cv2.waitKey(0)
        x[0] = np.transpose(imt, (2, 0, 1)).astype(np.float32) - self.mean
        for i in range(begin_num + stride, begin_num + self.T_len*stride, stride):

            rand_num = np.random.randint(0, self.n)
            # print(self.temp_distru)
            t_d = np.random.choice(self.temp_distru, 1, replace=False)[0]
            name = 'f{:05d}_d{:1.1f}_n{:01d}*'.format(i,
                                                      t_d,
                                                      rand_num)
            temp_pathN = glob.glob(join(self.root, folder_name, name))
            xy_bias = temp_pathN[0].replace(self.root+'/'+folder_name+'/', '')
            xy_bias = xy_bias.replace(
                'f{:05d}_d{:1.1f}_n{:01d}_'.format(i,
                                                   t_d,
                                                   rand_num),
                ''
            )
            xy_bias = xy_bias.replace('.jpg', '')
            xy_bias = xy_bias.replace('x', '')
            xy_bias = xy_bias.replace('y', '')
            x_bias_str = int(xy_bias.split('_')[0])
            y_bias_str = int(xy_bias.split('_')[1])

            x_bias[int((i-begin_num)/stride)] = [x_bias_str, y_bias_str]

            im = cv2.imread(temp_pathN[0])
            x[int((i-begin_num)/stride)] = np.transpose(im, (2, 0, 1)).astype(np.float32) - self.mean
            # cv2.imshow("1", im)
            # cv2.waitKey(0)
            # x

        for i in range(begin_num, begin_num + self.T_len*stride, stride):
            rand_num = np.random.randint(0, self.n)
            s_d = np.random.choice(self.search_distru, 1, replace=False)[0]
            name = 'f{:05d}_d{:1.1f}_n{:01d}*'.format(i,
                                                      s_d,
                                                      rand_num)
            temp_pathN = glob.glob(join(self.root, folder_name, name))
            xy_bias = temp_pathN[0].replace(self.root+'/'+folder_name+'/', '')
            xy_bias = xy_bias.replace(
                'f{:05d}_d{:1.1f}_n{:01d}_'.format(i,
                                                   s_d,
                                                   rand_num),
                ''
            )
            xy_bias = xy_bias.replace('.jpg', '')
            xy_bias = xy_bias.replace('x', '')
            xy_bias = xy_bias.replace('y', '')
            x_bias_str = int(xy_bias.split('_')[0])
            y_bias_str = int(xy_bias.split('_')[1])

            z_bias[int((i-begin_num)/stride)] = [x_bias_str, y_bias_str]

            im = cv2.imread(temp_pathN[0])
            z[int((i-begin_num)/stride)] = np.transpose(im, (2, 0, 1)).astype(np.float32) - self.mean

            if self.train:
                res[int((i-begin_num)/stride)] = gaussian_shaped_labels_bias(self.output_sigma,
                                                                  self.res_size,
                                                                  -z_bias[int((i-begin_num)/stride)])

        if self.train:
            return x, z, res
        else:
            return x, z

    def __len__(self):
        return len(self.snp_index) * self.n

#
if __name__ == '__main__':
    config = TrackerConfig()
    train, train_i, test, test_i = assign_train_test(config, temp_distru=[0.1, 0.3, 0.5])
    print(train_i)
    xt, zt, r = train[50]
    #xte, zte, r = test[1]
    # print(r.shape)
    # print(test_i)
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

        imxfft = torch.rfft(torch.tensor(xt[0], dtype=torch.float).unsqueeze(0), signal_ndim=2, onesided=True)

        imzfft = torch.rfft(torch.tensor(zt[i], dtype=torch.float).unsqueeze(0), signal_ndim=2, onesided=True)
        #print(imxfft.shape)


        def complex_mulconj(x, z):
            out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
            out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
            return torch.stack((out_real, out_imag), -1)


        def complex_mul(x, z):
            out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
            out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
            return torch.stack((out_real, out_imag), -1)


        y = gaussian_shaped_labels(config.output_sigma, config.img_input_size)
        yt = torch.Tensor(y)
        yf = torch.rfft(yt.view(1, 1, config.img_input_size[0], config.img_input_size[1]), signal_ndim=2)

        kzzf = torch.sum(torch.sum(imxfft ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(imzfft, imxfft), dim=1, keepdim=True)
        alphaf = yf / (kzzf + config.lambda0)  # very Ugly
        response_feature = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2).numpy()

        res_feature =(response_feature - np.ones_like(response_feature) * np.min(response_feature)) \
                     / (np.max(response_feature) - np.min(response_feature))



        y = complex_mulconj(imzfft, imxfft)
        print(y.shape)
        y = y.sum(dim=1, keepdim=True)
        response_im = torch.irfft(y, signal_ndim=2, onesided=True).numpy()
        response_im = response_im + train.mean
        print(response_im)
        #response_im = (response_im - np.ones_like(response_im) * np.min(response_im)) / (np.max(response_im) - np.min(response_im))



        cv2.imshow("1", res_feature[0].transpose(1, 2, 0))
        cv2.waitKey(0)
        # print(res_im[0].shape)
        cv2.imshow("1", response_im[0].transpose(1, 2, 0))
        cv2.waitKey(0)

