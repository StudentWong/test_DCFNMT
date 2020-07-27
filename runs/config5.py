import numpy as np
from pytorch_gpu_memory.gpu_memory_log import gpu_memory_log
import torch
from train.util import *
import json
from train.modules.NTM import NTM


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data





class TrackerConfig(object):

    data_root = '/home/lilium/caijihuzhuo/OTB_wh103_p1.0'
    # data_root = '/home/studentw/disk3/tracker/test_DCFNMT/OTB_wh103_p1.0'

    # module_config_path = "/home/studentw/disk3/tracker/test_DCFNMT/train/modules/module_config.json"
    module_config_path = "/home/lilium/caijihuzhuo/test_DCFNMT/train/modules/module_config.json"

    #save_path = '/home/studentw/disk3/tracker/test_DCFNMT/work5'
    save_path = '/home/lilium/caijihuzhuo/test_DCFNMT/work5'
    module_config_base = load_json(module_config_path)
    module_config = module_config_base["duke"]["default"]

    mult_model = False
    C_blur = True
    C_Erase = True
    C_blur_inherit = False
    direct_correlation = False
    C_norm = True
    norm_learnable = True
    long_term = True

    cnn_struct = module_config["cnn"]

    img_input_size = module_config["img_input_size"]
    w_CNN_out = cnn_struct[-2]['out_sizes'][0]
    h_CNN_out = cnn_struct[-2]['out_sizes'][1]
    net_input_size = [w_CNN_out, h_CNN_out]
    c_CNN_out = cnn_struct[-2]['out_features']
    CNN_padding = False
    key_feature_num = 8
    dim_h_o = c_CNN_out * key_feature_num
    dim_C2_1 = w_CNN_out * h_CNN_out
    dim_C2_2 = c_CNN_out

    use_apex = True
    apex_level = "O0"

    adjust_lr = False
    T = 15
    batch = 12
    data_use = 98
    lr = 5e-3
    epochs = 250
    weight_decay = 1e-6

    lambda0 = 1e-4
    padding = 1.0
    output_sigma_factor = 0.1
    output_sigma = img_input_size[0] / (1 + padding) * output_sigma_factor


    num_scale = 5
    # num_scale = 5
    scale_step = 1.0275
    # scale_step = 1.3
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    # scale_factor = scale_step ** (np.arange(num_scale) - int(num_scale / 2))
    scale_penalty = 0.9925
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))
    net_average_image = np.expand_dims(np.expand_dims(np.array([127, 127, 127]), axis=1), axis=1).astype(np.float32)
    y = gaussian_shaped_labels(output_sigma, [w_CNN_out, h_CNN_out])
    yt = torch.Tensor(y)
    #label_sum = yt.sum().cuda()
    yf = torch.rfft(yt.view(1, 1, w_CNN_out, h_CNN_out).cuda(), signal_ndim=2)
    #label_sum = label_sum.expand_as(yt.view(1, 1, w_CNN_out, h_CNN_out))
    # cos_window = torch.Tensor(np.outer(np.hanning(crop_sz_y), np.hanning(crop_sz_x))).cuda()

# if __name__ == "__main__":

    #
    #
    #
    #
    # o = TrackerConfig()
    # nmtcell = NTM(o,True).cuda()
    # C0 = torch.rand((o.batch, o.dim_C2_1, o.dim_C2_2), dtype=torch.float)*100
    # C0 = C0.cuda()
    # h0 = torch.rand((o.batch, o.dim_h_o), dtype=torch.float)*100
    # h0 = h0.cuda()
    #
    # CX = torch.rand((o.batch, o.T, o.dim_C2_1, o.dim_C2_2), dtype=torch.float)*1
    # CX = CX.cuda()
    #
    #
    # h,c = nmtcell.forward_batch(h0, C0, CX)

    # print(h)
    # print(h.shape)
    # print(c)
    # print(c.shape)
    # gpu_memory_log()

# class TrackerConfig(object):
#     # These are the default hyper-params for DCFNet
#     # OTB2013 / AUC(0.665)
#     feature_path = 'param.pth'
#     # crop_sz_x = 160
#     # crop_sz_y = 320
#     crop_sz = 125
#     train_step = 1
#     train_batch = 10
#     strong_weak_t = -0.2
#     lambda0 = 1e-4
#     padding = 2
#     output_sigma_factor = 0.1
#     feature_out_channel = 32
#     interp_factor = 0.01
#     num_scale = 5
#     scale_step = 1.1
#     scale_factor = scale_step ** (np.arange(num_scale) - int(num_scale / 2))
#     #scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
#     #print(scale_factor)
#     response_max_t = 0.2
#     module_update_lr = 0.1
#     #print(np.arange(num_scale) - int(num_scale / 2))
#     min_scale_factor = 0.2
#     max_scale_factor = 5
#     vanish_count_t = 50
#     scale_penalty = 0.9925
#     #scale_penalty = 0.8
#     scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - int(num_scale / 2))))
#     # print(scale_penalties)
#     #net_input_size = [crop_sz_y, crop_sz_x]
#     net_input_size = [crop_sz, crop_sz]
#     #net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
#     #output_sigma_x = crop_sz_x / (1 + padding) * output_sigma_factor
#     output_sigma = crop_sz / (1 + padding) * output_sigma_factor
#     #output_sigma_y = crop_sz_y / (1 + padding) * output_sigma_factor
#     #y = gaussian_shaped_labels(output_sigma_x, net_input_size)
#     y = gaussian_shaped_labels(output_sigma, net_input_size)
#     #yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz_y, crop_sz_x).cuda(), signal_ndim=2, onesided=False)
#     yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda(), signal_ndim=2)
#     #cos_window = torch.Tensor(np.outer(np.hanning(crop_sz_y), np.hanning(crop_sz_x))).cuda()
#     cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()
#
#     outputsize = [1, 1, net_input_size[0], net_input_size[1]]
#
#     regionregressionweightX = np.zeros(outputsize, dtype=np.float)
#     regionregressionweightY = np.zeros(outputsize, dtype=np.float)
#     regionregressionweightScale = np.zeros(outputsize, dtype=np.float)
#     for i in range(0, net_input_size[0]):
#         for j in range(0, net_input_size[1]):
#             regionregressionweightX[0][0][i][j] = i if i < net_input_size[0] / 2 else i - net_input_size[
#                                                                                                           0]
#             regionregressionweightY[0][0][i][j] = j if j < net_input_size[1] / 2 else j - net_input_size[
#                                                                                                           1]
#             #regionregressionweightScale[0][k][i][j] = scale_factor[k]
#             # regionregressionweightX[0][0][i][j] = i
#             # regionregressionweightY[0][0][i][j] = j
#
#     regionregressionweightX = torch.tensor(regionregressionweightX)
#     regionregressionweightX = torch.nn.Parameter(regionregressionweightX, requires_grad=False)
#
#     regionregressionweightY = torch.tensor(regionregressionweightY)
#     regionregressionweightY = torch.nn.Parameter(regionregressionweightY, requires_grad=False)
#
