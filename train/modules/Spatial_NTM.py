from torch import nn
import numpy as np
import torch
from train.modules.common import FCN, norm_grad, Identity, GaussianBlurConv
from torch.utils.checkpoint import checkpoint


# from train.config import TrackerConfig


# class NTM_pack(nn.Module):
#     def __init__(self, o):
#         super(NTM_pack, self).__init__()
#         self.o = o


class SpatialNTM(nn.Module):

    def __init__(self, config, use_checkpoint):
        super(SpatialNTM, self).__init__()
        self.config = config
        # fcn_params = [config.dim_h_o] + config.fcn + [dim_y]
        # self.fcn = FCN(fcn_params, hid_trans='relu', out_trans=None)
        # self.softmax = nn.Softmax(dim=1)
        self.use_checkpoint = use_checkpoint
        self.ntm_cell = Spatial_NTMCell_single(config, use_checkpoint)
        if self.config.C_blur:
            self.C_Blur = GaussianBlurConv(self.config.c_CNN_out)
        # self.att = torch.Tensor(config.batch, config.T, self.ntm_cell.ha, self.ntm_cell.wa).cuda()
        # self.mem = torch.Tensor(config.batch, config.T, self.ntm_cell.ha, self.ntm_cell.wa).cuda()

    def forward(self, h_o_prev, c_prev, x_input):
        """
        h_o_prev: N * dim_h_o
        c_prev: N * C2_1 * C2_2
        """
        h, c = self.ntm_cell(h_o_prev, c_prev, x_input)

        return h, c

    def forward_batch(self, h0, c0, c_x, ):
        """
        h0: N * dim_h_o
        c0: N * C2_1 * C2_2
        c_x: N * T * C2_1 * C2_2
        """
        # c = c0

        out_h = [h0]
        out_C = [c0]
        c_x_input = torch.unbind(c_x, dim=1)  # N * C2_1 * C2_2
        for t in range(0, self.config.T):
            h, c = self.ntm_cell(out_h[t], out_C[t], c_x_input[t])
            if self.config.C_blur:
                if self.config.C_blur_inherit:
                    c = c.permute(0, 2, 1).contiguous()
                    # print(c.is_contiguous())
                    assert c.is_contiguous(), "view not contiguous"
                    c = c.view(self.config.batch, self.config.dim_C2_2, self.config.w_CNN_out, self.config.h_CNN_out)
                    c = self.C_Blur(c)
                    assert c.is_contiguous(), "view not contiguous"
                    c = c.view(self.config.batch, self.config.dim_C2_2, self.config.dim_C2_1)
                    c = c.permute(0, 2, 1).contiguous()

            out_h.append(h)
            out_C.append(c)
        # print(len(out_C))
        # print(out_C[1].shape)
        if not self.config.long_term:
            first_return = torch.stack(out_h[1:], dim=1)
        else:
            first_return = out_h[-1].unsqueeze(1)

        if self.config.multi_C_output:
            second_return = torch.stack(out_C[1:], dim=1)
        else:
            second_return = out_C[-1].unsqueeze(1)

        return [first_return, second_return]


class Spatial_NTMCell_single(nn.Module):

    def __init__(self, config, use_checkpoint):
        super(Spatial_NTMCell_single, self).__init__()
        self.config = config
        self.bias_num = len(config.Spatial_Bias_List)
        self.use_checkpoint = use_checkpoint
        if not self.config.mult_model:
            self.linear_k = nn.Linear(config.dim_h_o, config.dim_C2_2)
            self.linear_b = nn.Linear(config.dim_h_o, 1)
            self.linear_e = nn.Linear(config.dim_h_o, config.dim_C2_2)
            self.linear_v = nn.Linear(config.dim_h_o, config.dim_C2_2)
            self.cosine_similarity = nn.CosineSimilarity(dim=2, eps=self.config.cos_esp)
            self.rnn_cell = nn.GRUCell(config.dim_C2_2 + self.bias_num, config.dim_h_o + self.bias_num)
            self.softmax = nn.Softmax(dim=1)
            self.softman_bias = nn.Softmax(dim=2)
        else:
            self.linear_k = nn.Linear(config.dim_h_o, config.dim_h_o)
            self.linear_b = nn.Linear(config.dim_h_o, self.config.key_feature_num)
            self.linear_e = nn.Linear(config.dim_h_o, config.dim_h_o)
            self.linear_v = nn.Linear(config.dim_h_o, config.dim_h_o)
            self.cosine_similarity = nn.CosineSimilarity(dim=3, eps=self.config.cos_esp)
            self.rnn_cell = nn.GRUCell((self.config.c_CNN_out + self.bias_num) * self.config.key_feature_num,
                                       (self.config.c_CNN_out + self.bias_num) * self.config.key_feature_num)
            self.softmax = nn.Softmax(dim=1)
            self.softman_bias = nn.Softmax(dim=3)

        if self.config.C_norm:
            self.norm_c = torch.nn.LayerNorm([config.dim_C2_1, config.dim_C2_2],
                                             elementwise_affine=config.norm_learnable)

        # self.ha = int(round(np.sqrt(config.dim_C2_1 * config.H / config.W)))
        # self.wa = int(round(np.sqrt(config.dim_C2_1 * config.W / config.H)))

        self.i = 0  # object id
        self.t = 0  # time
        self.n = 0  # sample id

    def forward(self, h_o_prev, c_prev, c_xf):
        return self.forward_no_checkpoint_full_version(h_o_prev, c_prev, c_xf)

    def forward_no_checkpoint_full_version(self, h_o_prev, c_prev, c_xf):
        """
        h_o_prev: N * dim_h_o
        C: N * C2_1 * C2_2
        """
        if not self.config.mult_model:
            # Addressing key
            # print(h_o_prev.shape)
            origin_h, bias = h_o_prev.split([self.config.dim_h_o, self.bias_num], dim=1)
            k = self.linear_k(origin_h)  # N * C2_2
            k_expand = k.unsqueeze(1).expand_as(c_prev)  # N * C2_1 * C2_2
            # Key strength, which equals to beta_pre.exp().log1p() + 1 but avoids 'inf' caused by exp()
            beta_pre = self.linear_b(origin_h)
            beta_pos = beta_pre.clamp(min=0)
            beta_neg = beta_pre.clamp(max=0)
            beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2))  # N * 1
            # Weighting
            # C_cos = Identity()(C)
            # norm_grad(h_o_prev, 1)
            norm_grad(c_xf, 1)
            # C2_2是通道数， 在余弦相似度中消失
            # print(C_cos.shape)
            # print(k_expand.shape)

            # Write vector and weight
            s_kc = self.cosine_similarity(c_prev, k_expand)
            assert s_kc.is_contiguous(), "view not contiguous"
            s_kc = s_kc.view(-1, self.config.dim_C2_1)  # N * C2_1
            w_kc = self.softmax(s_kc * beta)  # N * C2_1
            assert w_kc.is_contiguous(), "view not contiguous"
            w_kc_NWH = w_kc.view(self.config.batch, self.config.w_CNN_out, self.config.h_CNN_out)

            w_kc_bias = []
            for index in range(0, self.bias_num):
                w_kc_bias_ = w_kc_NWH.roll(self.config.Spatial_Bias_List[index], dims=(1, 2))
                w_kc_bias = w_kc_bias + [w_kc_bias_]
            w_bias_NWHB_todim = torch.stack(w_kc_bias, dim=3)
            assert w_bias_NWHB_todim.is_contiguous(), "view not contiguous"
            w_bias_todim = w_bias_NWHB_todim.view(self.config.batch, self.config.dim_C2_1, self.bias_num)
            w_bias_todim = self.softman_bias(w_bias_todim)

            # Read vector and weight
            s_kx = self.cosine_similarity(c_xf, k_expand)
            assert s_kx.is_contiguous(), "view not contiguous"
            s_kx = s_kx.view(-1, self.config.dim_C2_1)  # N * C2_1
            w_kx = self.softmax(s_kx * beta)  # N * C2_1
            w1 = w_kx.unsqueeze(1)  # N * 1 * C2_1
            norm_grad(w1, 1)

            c_with_bias = torch.cat([c_xf, w_bias_todim], dim=2)
            assert c_with_bias.is_contiguous(), "view not contiguous"
            # print(c_with_bias.shape)
            r = w1.bmm(c_with_bias).squeeze(1)  # N * C2_2
            # print(r)

            # RNN
            h_o = self.rnn_cell(r, h_o_prev)

            # if "no_mem" not in config.exp_config:
            if True:
                origin_o_h, h_o_bias = h_o.split([self.config.dim_h_o, self.bias_num], dim=1)
                h_o_bias = self.softmax(h_o_bias).unsqueeze(2)

                w_write = w_bias_todim.bmm(h_o_bias)
                # Erase vector
                e = self.linear_e(origin_o_h).sigmoid().unsqueeze(1)  # N * 1 * C2_2
                # Write vector
                v = self.linear_v(origin_o_h).unsqueeze(1)  # N * 1 * C2_2
                # Update memory
                w2 = w_kc.unsqueeze(2)  # N * C2_1 * 1
                if self.config.C_Erase:
                    we = w2.sum(dim=2, keepdim=True).expand_as(c_prev)
                    c_new = we * c_prev * (1 - w2.bmm(e)) + w_write.bmm(v)  # N * C2_1 * C2_2
                else:
                    c_new = c_prev * (1 - w2.bmm(e)) + w_write.bmm(v)
                # c_new = c_prev * (1 - w2.bmm(e)) + w_write.bmm(v)  # N * C2_1 * C2_2
                norm_grad(c_new, 1)

            # if config.v > 0:
            #     self.att[self.i].copy_(w.data[n].view(self.ha, self.wa))

            # return h_o, C, k, r

        else:
            # Addressing key
            origin_h, bias = h_o_prev.split([self.config.dim_h_o, self.bias_num*self.config.key_feature_num], dim=1)

            k = self.linear_k(origin_h)  # N * C2_2
            assert k.is_contiguous(), "not contiguous"
            k = k.view(self.config.batch, self.config.key_feature_num, self.config.dim_C2_2)  # N *key* C2_2
            k_expand = k.unsqueeze(2).expand(self.config.batch, self.config.key_feature_num,
                                             self.config.dim_C2_1, self.config.dim_C2_2)
            # N * core * C2_1 * C2_2

            # k_expand = k.unsqueeze(1).expand_as(c_prev)  # N * C2_1 * C2_2
            # Key strength, which equals to beta_pre.exp().log1p() + 1 but avoids 'inf' caused by exp()
            beta_pre = self.linear_b(origin_h)
            beta_pos = beta_pre.clamp(min=0)
            beta_neg = beta_pre.clamp(max=0)
            beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2))  # N * key_num
            beta = beta.unsqueeze(2).expand(self.config.batch, self.config.key_feature_num, self.config.dim_C2_1)
            beta = beta.contiguous()
            assert beta.is_contiguous(), "view not contiguous"
            beta = beta.view(self.config.batch, self.config.key_feature_num * self.config.dim_C2_1)
            # Weighting
            # C_cos = Identity()(C)
            # norm_grad(h_o_prev, 1)
            norm_grad(c_xf, 1)
            # C2_2是通道数， 在余弦相似度中消失

            # Write vector and weight
            c_prev_expand = c_prev.unsqueeze(1).expand((self.config.batch, self.config.key_feature_num,
                                                        self.config.dim_C2_1, self.config.dim_C2_2))
            s_kc = self.cosine_similarity(c_prev_expand, k_expand)
            assert s_kc.is_contiguous(), "view not contiguous"
            s_kc = s_kc.view(self.config.batch,
                             self.config.key_feature_num * self.config.dim_C2_1)  # N * (key_feature_num * C2_1)

            w_kc = self.softmax(s_kc * beta)  # N * (key_feature_num * C2_1)

            assert w_kc.is_contiguous(), "view not contiguous"
            w_kc = w_kc.view(self.config.batch,
                             self.config.key_feature_num, self.config.dim_C2_1)  # N * key_feature_num * C2_1

            assert w_kc.is_contiguous(), "view not contiguous"
            w_kc_NWH = w_kc.view(self.config.batch, self.config.key_feature_num,
                                 self.config.w_CNN_out, self.config.h_CNN_out)

            w_kc_bias = []
            for index in range(0, self.bias_num):
                w_kc_bias_ = w_kc_NWH.roll(self.config.Spatial_Bias_List[index], dims=(2, 3))
                w_kc_bias = w_kc_bias + [w_kc_bias_]
            w_bias_NWHB_todim = torch.stack(w_kc_bias, dim=4)
            assert w_bias_NWHB_todim.is_contiguous(), "view not contiguous"
            w_bias_todim = w_bias_NWHB_todim.view(self.config.batch, self.config.key_feature_num,
                                                  self.config.dim_C2_1, self.bias_num)
            w_bias_todim = self.softman_bias(w_bias_todim)

            # Read vector and weight
            # c_xf_expand = c_xf

            c_xf_expand = c_xf.unsqueeze(1).expand((self.config.batch, self.config.key_feature_num,
                                                    self.config.dim_C2_1, self.config.dim_C2_2))

            s_kx = self.cosine_similarity(c_xf_expand, k_expand)  # N * key_feature_num * C2_1

            assert s_kx.is_contiguous(), "view not contiguous"
            s_kx = s_kx.view(self.config.batch,
                             self.config.key_feature_num * self.config.dim_C2_1)  # N * C2_1 * key_feature_num
            w_kx = self.softmax(s_kx * beta)  # N * (key_feature*C2_1)
            # assert w_kx.is_contiguous(), "view not contiguous"
            # w_kx = w_kx.view(self.config.batch, self.config.key_feature_num, self.config.dim_C2_1)
            assert w_kx.is_contiguous(), "view not contiguous"
            w_kx = w_kx.view(self.config.batch * self.config.key_feature_num, self.config.dim_C2_1)
            w1 = w_kx.unsqueeze(1)  # N * 1 * C2_1
            norm_grad(w1, 1)
            c_xf_expand = c_xf_expand.contiguous()
            assert c_xf_expand.is_contiguous(), "view not contiguous"
            c_xf_expand = c_xf_expand.view(self.config.batch * self.config.key_feature_num,
                                           self.config.dim_C2_1, self.config.dim_C2_2)

            assert w_bias_todim.is_contiguous(), "view not contiguous"
            w_bias_todim = w_bias_todim.view(self.config.batch * self.config.key_feature_num,
                                             self.config.dim_C2_1, self.bias_num)

            c_with_bias = torch.cat([c_xf_expand, w_bias_todim], dim=2)
            assert c_with_bias.is_contiguous(), "view not contiguous"

            r = w1.bmm(c_with_bias).squeeze(1)  # N * C2_2
            assert r.is_contiguous(), "view not contiguous"
            # print(r.shape)
            r = r.view(self.config.batch, self.config.key_feature_num * (self.config.dim_C2_2+self.bias_num))

            # print(w_kc.shape)
            # RNN
            h_o = self.rnn_cell(r, h_o_prev)

            if True:
                # Erase vector
                origin_h_o, h_o_bias = h_o.split([self.config.dim_h_o, self.bias_num * self.config.key_feature_num],
                                                dim=1)
                # print(h_o_bias.shape)
                h_o_bias = h_o_bias.contiguous()
                assert h_o_bias.is_contiguous(), "view not contiguous"
                h_o_bias = h_o_bias.view(self.config.batch * self.config.key_feature_num, self.bias_num)
                # print(w_bias_todim.shape)
                # exit()
                h_o_bias = self.softmax(h_o_bias).unsqueeze(2)

                w_write = w_bias_todim.bmm(h_o_bias)
                assert w_write.is_contiguous(), "view not contiguous"
                w_write = w_write.view(self.config.batch, self.config.key_feature_num, self.config.dim_C2_1)
                w_write = w_write.permute(0, 2, 1).contiguous()

                e = self.linear_e(origin_h_o).sigmoid()
                assert e.is_contiguous(), "view not contiguous"
                e = e.view(self.config.batch, self.config.key_feature_num, self.config.dim_C2_2)  # N * key * C2_2
                # Write vector
                v = self.linear_v(origin_h_o)  # N * 1 * C2_2
                assert v.is_contiguous(), "view not contiguous"
                v = v.view(self.config.batch, self.config.key_feature_num, self.config.dim_C2_2)  # N * key * C2_2

                # Update memory
                w2 = w_kc.permute(0, 2, 1).contiguous()  # N * C2_1 * key_feature_num
                assert w2.is_contiguous(), "view not contiguous"
                if self.config.C_Erase:
                    we = w2.sum(dim=2, keepdim=True).expand_as(c_prev)
                    c_new = we * c_prev * (1 - w2.bmm(e)) + w_write.bmm(v)  # N * C2_1 * C2_2
                else:
                    c_new = c_prev * (1 - w2.bmm(e)) + w_write.bmm(v)
                norm_grad(c_new, 1)
        if self.config.C_norm:
            c_new = self.norm_c(c_new)
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


if __name__ == "__main__":
    o = TrackerConfig()
    # ntm = NTMCell(o)
