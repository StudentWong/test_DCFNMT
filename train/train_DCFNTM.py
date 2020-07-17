from torch import nn
from os.path import join, isfile, isdir
from os import makedirs
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
from train.config import TrackerConfig
import time
import argparse

parser = argparse.ArgumentParser(description='Training DCFNet in Pytorch 0.4.0')
parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
parser.add_argument('--padding', dest='padding', default=1.0, type=float, help='crop padding size')
parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 5e-5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

args = parser.parse_args()
train_loss = []
val_loss = []

print(args)
best_loss = 1e6

config = TrackerConfig()

model = DCFNTM(config, True)
model.cuda()
gpu_num = torch.cuda.device_count()
print('GPU NUM: {:2d}'.format(gpu_num))
if gpu_num > 1:
    model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()

criterion = nn.MSELoss(size_average=False).cuda()
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

save_path = join(args.save, 'T{:d}_b{:d}_C{:d}_data{:d}_lr{:.3e}'.format(config.T, config.batch,
                                                                         config.dim_C2_2, config.data_use,
                                                                         config.lr))
if not isdir(save_path):
    makedirs(save_path)

train_dataset, t_i, val_dataset, v_i = assign_train_test(config)

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
    for i, (template, search, response) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        template = template.cuda(non_blocking=True)
        search = search.cuda(non_blocking=True)
        response = response.cuda(non_blocking=True)

        # compute output
        output = model(template, search)
        # print(output.shape)
        # print(response.shape)
        loss = criterion(output, response) / template.size(0)  # criterion = nn.MSEloss

        # measure accuracy and record loss
        losses.update(loss.item())

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
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
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
        for i, (template, search, response) in enumerate(val_loader):

            # compute output
            template = template.cuda(non_blocking=True)
            search = search.cuda(non_blocking=True)
            response = response.cuda(non_blocking=True)

            # compute output
            output = model(template, search)
            loss = criterion(output, response) / (args.batch_size * gpu_num)

            # measure accuracy and record loss
            losses.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))

        val_loss_plot = val_loss_plot + [losses.avg]
        print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

    return losses.avg, val_loss_plot


for epoch in range(args.start_epoch, args.epochs):
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
plt.savefig(join(args.save, 'train_T{:d}_b{:d}_C{:d}_data{:d}_lr{:.3e}.jpg'.format(config.T, config.batch,
                                                                               config.dim_C2_2, config.data_use,
                                                                               config.lr)))

plt.figure(1)
plt.plot(np.array(val_loss))
plt.savefig(join(args.save, 'val_T{:d}_b{:d}_C{:d}_data{:d}_lr{:.3e}.jpg'.format(config.T, config.batch,
                                                                               config.dim_C2_2, config.data_use,
                                                                               config.lr)))
# plt.show()
