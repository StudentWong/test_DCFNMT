import torch
import torch.nn as nn


class LossCalculator(nn.Module):

    def __init__(self, config):
        super(LossCalculator, self).__init__()
        self.config = config
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_C_predict = self.config.lambda_C_predict

        # self.bce_loss = nn.BCELoss(size_average=False)

    def response_loss(self, output, response):
        responseloss = self.mse(output, response) / (response.shape[0] * response.shape[1])
        return responseloss

    def C_predict_loss(self, cf, zf):
        Cpredictloss = self.mse(cf, zf) / (cf.shape[0] * cf.shape[1])
        return Cpredictloss

    def C_depress_loss(self, cf):
        '''
        :param cf: N * T * C2_1 * C2_2
        :return:
        '''
        two_norm = torch.norm(cf, dim=3)
        one_norm = torch.norm(two_norm, dim=2, p=1)
        Cdepressloss = torch.sum(one_norm)
        return Cdepressloss


# if __name__ == "__main__":

