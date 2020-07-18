from torch import nn
import torch
# from torchsummary import summary
from train.config import TrackerConfig


def act_func(func_name):
    if func_name is None:
        return None
    elif func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU(inplace=True)
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    else:
        assert False, 'Invalid func_name.'

def norm_func(func_name):
    if func_name is None:
        return None
    elif func_name == 'LRN':
        return nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
    else:
        assert False, 'Invalid func_name.'


class Feature(nn.Module):
    def __init__(self, config):
        super(Feature, self).__init__()
        self.config = config

        for i in range(0, len(config.cnn_struct) - 1):
            if config.CNN_padding:
                padding = int((config.cnn_struct[i]["conv_kernels"] - 1) / 2)
            else:
                padding = 0
            setattr(self, 'conv' + str(i),
                    nn.Conv2d(config.cnn_struct[i]["in_features"],
                              config.cnn_struct[i]["out_features"],
                              config.cnn_struct[i]["conv_kernels"],
                              padding=padding)
                    )

            if config.cnn_struct[i]["bn"] != -1:
                setattr(self, 'bn' + str(i), nn.BatchNorm2d(config.cnn_struct[i]["out_features"]))

            if config.cnn_struct[i]["dropout"] != -1:
                setattr(self, 'dp' + str(i), nn.Dropout2d(0.2, inplace=True))

            if config.cnn_struct[i]["activate_fun"] != "none":
                setattr(self, 'act' + str(i), act_func(config.cnn_struct[i]["activate_fun"]))

        if config.cnn_struct[-1]["norm"] != "none":
            setattr(self, 'norm', norm_func(config.cnn_struct[-1]["norm"]))

    def forward(self, x):
        h = x  # N * D * H * W
        for i in range(0, len(self.config.cnn_struct) - 1):
            h = getattr(self, 'conv' + str(i))(h)
            if self.config.cnn_struct[i]["bn"] != -1:
                h = getattr(self, 'bn' + str(i))(h)

            if self.config.cnn_struct[i]["dropout"] != -1:
                h = getattr(self, 'dp' + str(i))(h)

            if self.config.cnn_struct[i]["activate_fun"] != "none":
                h = getattr(self, 'act' + str(i))(h)

        if self.config.cnn_struct[-1]["norm"] != "none":
            h = getattr(self, 'norm')(h)

        return h


if __name__ == '__main__':
    config = TrackerConfig()
    net = Feature(config)
    #net = DCFNetFeature()
    # summary(net, input_size=(3, 99, 99))
    x = torch.ones((1, 3, 103, 103))*100
    y = net(x)
    y.mean().backward()

    for name, parms in net.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
              ' -->grad_value:', parms.grad)
    # print(y.shape)
