from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import FairSupConResNet
from losses import FairSupConLoss
from dataset import UTKLoader, UTKLoader, CelebaLoader

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='celeba',
                        choices=['celeba','utkface'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    
    parser.add_argument('--size', type=int, default=128, help='parameter for RandomResizedCrop')
    parser.add_argument('--name', type=str, default='',help='saved filename')
    parser.add_argument('--ckpt', type=str, default='', help='path to pre-trained model')
    

    # method
    parser.add_argument('--method', type=str, default='FSCL',
                        choices=['FSCL','FSCL*','SupCon', 'SimCLR'], help='choose method')
    #norm
    parser.add_argument('--group_norm', type=int, default=0, help='group normalization')
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    

    # attribute
    parser.add_argument('--target_attribute_1', type=str, default='',help='target attribute')
    parser.add_argument('--target_attribute_2', type=str, default='None',help='target attribute')
    parser.add_argument('--sensitive_attribute_1', type=str, default='',help='sensitive_attribute')
    parser.add_argument('--sensitive_attribute_2', type=str, default='None',help='sensitive_attribute')


    opt = parser.parse_args()
    # check if dataset is path that passed required arguments
    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/FairSupCon/{}_models'.format(opt.dataset)
  
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate


    opt.save_folder = os.path.join(opt.model_path, opt.model_name,opt.name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):

    mean = (0.5000, 0.5000, 0.5000)
    std = (0.5000, 0.5000, 0.5000)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'utkface':
        train_dataset = UTKLoader(0,ta=opt.target_attribute_1,sa=opt.sensitive_attribute_1,data_folder=opt.data_folder,transform=TwoCropTransform(train_transform))
    
    elif opt.dataset == 'celeba':
        train_dataset = CelebaLoader(0,ta=opt.target_attribute_1,ta2=opt.target_attribute_2,sa=opt.sensitive_attribute_1,sa2=opt.sensitive_attribute_2,data_folder=opt.data_folder,transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)

    return train_loader


def set_model(opt):
    model = FairSupConResNet(name=opt.model)
    criterion = FairSupConLoss(temperature=opt.temp)
    s_epoch=0
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if opt.ckpt!='':
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder)
            else:
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    new_state_dict[k] = v
                state_dict = new_state_dict
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

            model.load_state_dict(state_dict)
            s_epoch=ckpt['epoch']
    else:

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder)
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

    return model, criterion,s_epoch


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()

    for idx, (images,ta,sa) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            ta = ta.cuda(non_blocking=True)
            sa = sa.cuda(non_blocking=True)
        bsz = ta.shape[0]
      
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        
        features = model(images)
        
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)




        loss = criterion(features,ta,sa,opt.group_norm,opt.method,epoch)

    
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
    
    return losses.avg

def main():
    opt = parse_option()
    
    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    s_epoch=0
    model, criterion,s_epoch = set_model(opt)
    
    # build optimizer
    optimizer = set_optimizer(opt, model)


    
    # training routine
    for epoch in range(s_epoch+1, opt.epochs + 1):
        
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

     

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    
    main()
    