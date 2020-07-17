from torch import nn
import numpy as np
import torch
from train.modules.common import FCN, norm_grad, Identity
from torch.utils.checkpoint import checkpoint
# class NTM_pack(nn.Module):
#     def __init__(self, o):
#         super(NTM_pack, self).__init__()
#         self.o = o


class NTM(nn.Module):

    def __init__(self, config, use_checkpoint):
        super(NTM, self).__init__()
        self.config = config
        # fcn_params = [config.dim_h_o] + config.fcn + [dim_y]
        # self.fcn = FCN(fcn_params, hid_trans='relu', out_trans=None)
        # self.softmax = nn.Softmax(dim=1)
        self.use_checkpoint = use_checkpoint
        self.ntm_cell = NTMCell_single(config, use_checkpoint)
        # self.att = torch.Tensor(config.batch, config.T, self.ntm_cell.ha, self.ntm_cell.wa).cuda()
        # self.mem = torch.Tensor(config.batch, config.T, self.ntm_cell.ha, self.ntm_cell.wa).cuda()

    def forward(self,  h_o_prev, C):
        """
        h_o_prev: N * T * dim_h_o
        C: N * T * C2_1 * C2_2
        """
        C_t = torch.unbind(C, dim=1)  # N * C2_1 * C2_2
        h_o_prev_t = torch.unbind(h_o_prev, dim=1)  # N * dim_h_o
        out_h = []
        out_C = []
        for t in range(0, self.config.T):
            h, c = self.ntm_cell(h_o_prev_t[t], C_t[t])
            out_h.append(h)
            out_C.append(c)
        # print(len(out_C))
        # print(out_C[1].shape)
        return [torch.stack(out_h, dim=1), torch.stack(out_C, dim=1)]


    def forward_batch(self, h0, c0, c_x):
        """
        h0: N * dim_h_o
        c0: N * C2_1 * C2_2
        c_x: N * T * C2_1 * C2_2
        """
        #c = c0

        out_h = [h0]
        out_C = [c0]
        c_x_input = torch.unbind(c_x, dim=1)  # N * C2_1 * C2_2
        for t in range(0, self.config.T):
            h, c = self.ntm_cell(out_h[t], out_C[t], c_x_input[t])
            out_h.append(h)
            out_C.append(c)
        # print(len(out_C))
        # print(out_C[1].shape)
        return [torch.stack(out_h[1:], dim=1), torch.stack(out_C[1:], dim=1)]


class NTMCell_single(nn.Module):

    def __init__(self, config, use_checkpoint):
        super(NTMCell_single, self).__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint
        self.linear_k = nn.Linear(config.dim_h_o, config.dim_C2_2)
        self.linear_b = nn.Linear(config.dim_h_o, 1)
        self.linear_e = nn.Linear(config.dim_h_o, config.dim_C2_2)
        self.linear_v = nn.Linear(config.dim_h_o, config.dim_C2_2)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)
        self.rnn_cell = nn.GRUCell(config.dim_C2_2, config.dim_h_o)
        # self.ha = int(round(np.sqrt(config.dim_C2_1 * config.H / config.W)))
        # self.wa = int(round(np.sqrt(config.dim_C2_1 * config.W / config.H)))

        self.i = 0  # object id
        self.t = 0  # time
        self.n = 0  # sample id

    def checkpoint_seg1_k(self, h_o_prev):
        k = self.linear_k(h_o_prev)  # N * C2_2
        return k

    def checkpoint_seg1_b(self, h_o_prev):
        beta_pre = self.linear_b(h_o_prev)
        return beta_pre

    def checkpoint_seg1_e(self, h_o):
        e = self.linear_e(h_o).sigmoid().unsqueeze(1)  # N * 1 * C2_2
        return e

    def checkpoint_seg1_v(self, h_o):
        # Write vector
        v = self.linear_v(h_o).unsqueeze(1)  # N * 1 * C2_2
        return v

    def checkpoint_seg1_gru(self, r, h_o_prev):
        h_o = self.rnn_cell(r, h_o_prev)
        return h_o

    def forward(self, h_o_prev, c_prev, c_xf):
        return self.forward_no_checkpoint_full_version(h_o_prev, c_prev, c_xf)

    def forward_no_checkpoint_full_version(self, h_o_prev, c_prev, c_xf):
        """
        h_o_prev: N * dim_h_o
        C: N * C2_1 * C2_2
        """
        # if config.v > 0:
        #     if self.i == 0:
        #         self.att.fill_(0.5)
        #         self.mem.fill_(0.5)
        #     self.mem[self.i].copy_(C.data[n].mean(1).view(self.ha, self.wa))

        # Addressing key
        #print(h_o_prev.shape)
        k = self.linear_k(h_o_prev)  # N * C2_2
        k_expand = k.unsqueeze(1).expand_as(c_prev)  # N * C2_1 * C2_2
        # Key strength, which equals to beta_pre.exp().log1p() + 1 but avoids 'inf' caused by exp()
        beta_pre = self.linear_b(h_o_prev)
        beta_pos = beta_pre.clamp(min=0)
        beta_neg = beta_pre.clamp(max=0)
        beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2))  # N * 1
        # Weighting
        #C_cos = Identity()(C)

        norm_grad(c_xf, 1)
        # C2_2是通道数， 在余弦相似度中消失
        #print(C_cos.shape)
        #print(k_expand.shape)
        s = self.cosine_similarity(c_xf, k_expand)
        assert s.is_contiguous(), "view not contiguous"
        s = s.view(-1, self.config.dim_C2_1)  # N * C2_1
        w = self.softmax(s * beta)  # N * C2_1

        # Read vector
        w1 = w.unsqueeze(1)  # N * 1 * C2_1
        norm_grad(w1, 1)
        r = w1.bmm(c_xf).squeeze(1)  # N * C2_2
        # RNN
        h_o = self.rnn_cell(r, h_o_prev)

        # if "no_mem" not in config.exp_config:
        if True:
            # Erase vector
            e = self.linear_e(h_o).sigmoid().unsqueeze(1)  # N * 1 * C2_2
            # Write vector
            v = self.linear_v(h_o).unsqueeze(1)  # N * 1 * C2_2
            # Update memory
            w2 = w.unsqueeze(2)  # N * C2_1 * 1
            c_new = c_prev * (1 - w2.bmm(e)) + w2.bmm(v)  # N * C2_1 * C2_2
            norm_grad(c_new, 1)

        # if config.v > 0:
        #     self.att[self.i].copy_(w.data[n].view(self.ha, self.wa))

        # return h_o, C, k, r
        return h_o, c_new


# class NTMCell(nn.Module):
#
#     def __init__(self, o):
#         super(NTMCell, self).__init__()
#         self.o = o
#         self.linear_k = nn.Linear(o.dim_h_o, o.dim_h_o)
#         self.linear_b = nn.Linear(o.dim_h_o, 1)
#         #o.dim_C2_2
#         self.linear_e = nn.Linear(o.dim_h_o, o.dim_C2_2)
#         self.linear_v = nn.Linear(o.dim_h_o, o.dim_C2_2)
#         self.cosine_similarity = nn.CosineSimilarity(dim=2)
#         self.softmax = nn.Softmax(dim=1)
#         self.rnn_cell = nn.GRUCell(o.dim_C2_2, o.dim_h_o)
#         # self.ha = int(round(np.sqrt(o.dim_C2_1 * o.H / o.W)))
#         # self.wa = int(round(np.sqrt(o.dim_C2_1 * o.W / o.H)))
#
#         self.i = 0  # object id
#         self.t = 0  # time
#         self.n = 0  # sample id
#
#     def forward(self, h_o_prev, C):
#         """
#         h_o_prev: N * dim_h_o
#         C: N * C2_1 * C2_2
#         """
#         o = self.o
#         n = self.n
#
#         # if o.v > 0:
#         #     if self.i == 0:
#         #         self.att.fill_(0.5)
#         #         self.mem.fill_(0.5)
#         #     self.mem[self.i].copy_(C.data[n].mean(1).view(self.ha, self.wa))
#
#         # Addressing key
#         #print(h_o_prev.shape)
#         k = self.linear_k(h_o_prev)  # N * C2_2
#         k_expand = k.unsqueeze(1).expand_as(C)  # N * C2_1 * C2_2
#         # Key strength, which equals to beta_pre.exp().log1p() + 1 but avoids 'inf' caused by exp()
#         beta_pre = self.linear_b(h_o_prev)
#         beta_pos = beta_pre.clamp(min=0)
#         beta_neg = beta_pre.clamp(max=0)
#         beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2))  # N * 1
#         # Weighting
#         C_cos = Identity()(C)
#         norm_grad(C_cos, 1)
#         # C2_2是通道数， 在余弦相似度中消失
#         print(C_cos.shape)
#         print(k_expand.shape)
#         s = self.cosine_similarity(C_cos, k_expand).view(-1, o.dim_C2_1)  # N * C2_1
#         w = self.softmax(s * beta)  # N * C2_1
#
#         # Read vector
#         w1 = w.unsqueeze(1)  # N * 1 * C2_1
#         norm_grad(w1, 1)
#         r = w1.bmm(C).squeeze(1)  # N * C2_2
#         # RNN
#         h_o = self.rnn_cell(r, h_o_prev)
#
#         # if "no_mem" not in o.exp_config:
#         if True:
#             # Erase vector
#             e = self.linear_e(h_o).sigmoid().unsqueeze(1)  # N * 1 * C2_2
#             # Write vector
#             v = self.linear_v(h_o).unsqueeze(1)  # N * 1 * C2_2
#             # Update memory
#             w2 = w.unsqueeze(2)  # N * C2_1 * 1
#             C = C * (1 - w2.bmm(e)) + w2.bmm(v)  # N * C2_1 * C2_2
#
#         # if o.v > 0:
#         #     self.att[self.i].copy_(w.data[n].view(self.ha, self.wa))
#
#         return h_o, C, k, r

# class TrackerConfig(object):
#     w_CNN_out = 3
#     h_CNN_out = 3
#     c_CNN_out = 2
#     key_feature_num = 3
#     dim_h_o = c_CNN_out * key_feature_num
#     dim_C2_1 = w_CNN_out * h_CNN_out
#     dim_C2_2 = c_CNN_out
#     H = 128
#     W = 128
#     T = 10
#     batch = 80


# if __name__ == "__main__":
#     o = TrackerConfig()
#     ntm = NTMCell(o)