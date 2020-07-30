from torch import nn
import os
#os.chdir('/home/lilium/caijihuzhuo/test_DCFNMT')
from os.path import join, isfile, isdir
from os import makedirs
from train.loss_calculator import LossCalculator
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from train.dataprepare.data import assign_train_test
import torch
from train.modules.NTM import NTM
from train.modules.feature import Feature
import torch.backends.cudnn as cudnn
from apex import amp
from train.net import DCFNTM
#from train.config import TrackerConfig
import time
import argparse

parser = argparse.ArgumentParser(description='Training DCFNet in Pytorch 0.4.0')
#parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
#parser.add_argument('--padding', dest='padding', default=1.0, type=float, help='crop padding size')
#parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
#parser.add_argument('--epochs', default=50, type=int, metavar='N',
#                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
# parser.add_argument('-b', '--batch-size', default=32, type=int,
#                     metavar='N', help='mini-batch size (default: 32)')
# parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
#                     metavar='W', help='weight decay (default: 5e-5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', '-c', default='train.config.TrackerConfig', type=str, help='directory for config')

args = parser.parse_args()
train_loss = []
val_loss = []

print(args)
best_loss = 1e6

import importlib

s_path = args.config
s_path_name, s_class_name = s_path.rsplit('.', 1)
o_module = importlib.import_module(s_path_name)

CClass = getattr(o_module, s_class_name) # 获取对应的class
o_obj_0 = CClass() # 实例化
config = getattr(o_module, s_class_name)() #  获取class’后直接实例化




#config = TrackerConfig()

model = DCFNTM(config, True)
model.cuda()
gpu_num = torch.cuda.device_count()
print('GPU NUM: {:2d}'.format(gpu_num))
if gpu_num > 1:
    model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()

criterion = LossCalculator(config).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.99),
                             weight_decay=config.weight_decay)

if config.use_apex:
    model, optimizer = amp.initialize(model, optimizer, opt_level=config.apex_level)

# optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)

# target = torch.Tensor(config.y).cuda().unsqueeze(0).unsqueeze(0).repeat(config.batch * gpu_num, 1, 1,
#                                                                         1)  # for training

if args.resume:
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# save_path = join(args.save, 'T{:d}_b{:d}_C{:d}_data{:d}_lr{:.3e}'.format(config.T, config.batch,
#                                                                          config.dim_C2_2, config.data_use,
#                                                                          config.lr))

save_path = config.save_path
if not isdir(save_path):
    makedirs(save_path)

train_dataset, t_i, val_dataset, v_i = assign_train_test(config)
print(v_i)

train_loader = DataLoader(
    train_dataset, batch_size=config.batch * gpu_num, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)

val_loader = DataLoader(
    val_dataset, batch_size=config.batch * gpu_num, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=True)


# cudnn.benchmark = True


def adjust_learning_rate(optimizer, epoch):
    lr = np.logspace(-2, -5, num=config.epochs)[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename=join(save_path, 'checkpoint.pth.tar')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(save_path, 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, train_loss_plot):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (template, search, response, ztemp) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        template = template.cuda(non_blocking=True).requires_grad_(True)
        search = search.cuda(non_blocking=True).requires_grad_(True)
        response = response.cuda(non_blocking=True).requires_grad_(True)

        # compute output
        if config.long_term:
            if config.multi_C_output:
                output, c, c_hidden = model(template, search)
            else:
                output, c = model(template, search)
        else:
            if config.multi_C_output:
                output, c_hidden = model(template, search)
            else:
                output = model(template, search)
        # print(output.shape)
        # print(response.shape)
        loss_response = criterion.response_loss(output, response)

        loss = loss_response.clone()

        if config.C_predict_loss:
            assert type(ztemp) == torch.Tensor, "error"
            ztemp = ztemp.cuda(non_blocking=True).requires_grad_(True)
            zf = model.CNN_Z(ztemp)
            loss = loss + config.lambda_C_predict*criterion.C_predict_loss(c_hidden, zf)

        if config.C_depress_loss:
            loss = loss + config.lambda_C_depress*criterion.C_depress_loss(c_hidden)
        # measure accuracy and record loss
        losses.update(loss_response.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if config.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] SumLoss {3:.4f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'ResponseLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), loss.item(), batch_time=batch_time,
                data_time=data_time, loss=losses))

            train_loss_plot = train_loss_plot + [losses.val]
    return train_loss_plot


def validate(val_loader, model, criterion, val_loss_plot):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (template, search, response, ztemp) in enumerate(val_loader):

            # compute output
            template = template.cuda(non_blocking=True).requires_grad_(True)
            search = search.cuda(non_blocking=True).requires_grad_(True)
            response = response.cuda(non_blocking=True).requires_grad_(True)

            # compute output
            if config.long_term:
                if config.multi_C_output:
                    output, c, c_hidden = model(template, search)
                else:
                    output, c = model(template, search)
            else:
                if config.multi_C_output:
                    output, c_hidden = model(template, search)
                else:
                    output = model(template, search)

            loss_response = criterion.response_loss(output, response)

            loss = loss_response.clone()

            if config.C_predict_loss:
                assert type(ztemp) == torch.Tensor, "error"
                ztemp = ztemp.cuda(non_blocking=True).requires_grad_(True)
                zf = model.CNN_Z(ztemp)
                loss = loss + config.lambda_C_predict * criterion.C_predict_loss(c_hidden, zf)

            if config.C_depress_loss:
                loss = loss + config.lambda_C_depress * criterion.C_depress_loss(c_hidden)

            # measure accuracy and record loss
            losses.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}] SumLoss {loss.val:.4f} ({loss.avg:.4f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'ResponseLoss {2:.4f})\t'.format(
                    i, len(val_loader), loss_response.item(), batch_time=batch_time, loss=losses))

        val_loss_plot = val_loss_plot + [losses.avg]
        print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

    return losses.avg, val_loss_plot


for epoch in range(args.start_epoch, config.epochs):
    if config.adjust_lr:
        adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train_loss = train(train_loader, model, criterion, optimizer, epoch, train_loss)

    # evaluate on validation set
    loss, val_loss = validate(val_loader, model, criterion, val_loss)

    # remember best loss and save checkpoint
    is_best = loss < best_loss
    best_loss = min(best_loss, loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)

plt.figure(0)
plt.plot(np.array(train_loss))
plt.savefig(join(save_path, 'train_T{:d}_b{:d}_C{:d}_data{:d}_lr{:.3e}.jpg'.format(config.T, config.batch,
                                                                               config.dim_C2_2, config.data_use,
                                                                               config.lr)))

plt.figure(1)
plt.plot(np.array(val_loss))
plt.savefig(join(save_path, 'val_T{:d}_b{:d}_C{:d}_data{:d}_lr{:.3e}.jpg'.format(config.T, config.batch,
                                                                               config.dim_C2_2, config.data_use,
                                                                               config.lr)))
# plt.show()
