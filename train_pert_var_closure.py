''' this script works with CPU and GPUs
'''
from __future__ import print_function

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import random
import numpy as np
import scipy.io

import argparse
import os
import gflags
import sys

from utils import Logger, AverageMeter
import models
import l96_sys

# hyper parameters for training
parser = argparse.ArgumentParser(description='model configuration')
# data loading parameters
parser.add_argument('--train_length', default=500, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--pred_length', default=5000, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--input_length', default=100, type=int, metavar='L',
                    help='model input state size')
parser.add_argument('--iters', default=100, type=int, metavar='I',
                    help='number of iterations for each epoch')
parser.add_argument('--train-batch', default=100, type=int, metavar='B',
                    help='each training batch size')
parser.add_argument('--nskip', default=10, type=int, metavar='nk',
                    help='time step skip in the loaded raw data, dt=1e-2')
parser.add_argument('--npred', default=5, type=int, metavar='Np',
                    help='number of iterations to measure in the loss func.')
# (train-batch * iters = train_length - input_length - npred+1)

# model parameters
parser.add_argument('--epoch', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nhid', default=100, type=int, metavar='nh',
                    help='hidden layer size in the network cell')
parser.add_argument('--nloss', default=50, type=int, metavar='nv',
                    help='number of steps to measure in the loss function')
# optimization parameters
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25, 40],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.5, 
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--loss-type', '--lt', default='flux', type=str, metavar='LT',
                    help='choices of loss functions (state,flux,comb, kld,mixed)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/train_vparam_F8pert', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file1', default='data/l96_nt1_fpert_F8amp1', type=str, metavar='DATA_PATH',
                    help='path to data set ramp up')
parser.add_argument('--data-file2', default='data/l96_nt1_upert_F8amp1', type=str, metavar='DATA_PATH',
                    help='path to data set ramp_down')
parser.add_argument('--data-type', '--dp', default='pertfu', type=str, metavar='DP',
                    help='choices of loss functions (pertf, pertu, pertfu)')
parser.add_argument('--pred-file1', default='data/l96_nt1_ramp1_F8df1', type=str, metavar='DATA_PATH',
                    help='path to data set ramp up')
parser.add_argument('--pred-file2', default='data/l96_nt1_ramp2_F8df1', type=str, metavar='DATA_PATH',
                    help='path to data set ramp_down')
parser.add_argument('--pred-file3', default='data/l96_nt1_peri_F8df1', type=str, metavar='DATA_PATH',
                    help='path to data set ramp_down')
parser.add_argument('--pred-file4', default='data/l96_nt1_peri_F8df2', type=str, metavar='DATA_PATH',
                    help='path to data set ramp_down')
parser.add_argument('--resume', default=False, type=bool, metavar='R_PATH',
                    help='path to latest checkpoint (default: none)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
cfg = {k: v for k, v in args._get_kwargs()}

if not os.path.isdir(args.checkpoint):
    os.makedirs(args.checkpoint)
        
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

# save config
fname = 'vclos_{}_adamlr1_ls1stg0hs{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.data_type, args.nhid, args.nloss,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
with open(args.checkpoint + "/config_"+fname+".txt", 'w') as f:
    for (k, v) in args._get_kwargs():
        f.write(k + ' : ' + str(v) + '\n')
    f.write('\n')

def main(pretrained = False, valid = False):
    # models for unresolved processes
    model = models.LSTMresi(input_size = 23, hidden_size = args.nhid, output_size = 21, 
                                nlayers = 1, nstages = 0).double()
    # load model on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('This code is run by {}: {} GPU(s)'.format(device, torch.cuda.device_count()))
    # if  torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model).to(dev)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    if pretrained:
        # load the pretrained model
        model_path = torch.load(os.path.join(cfg['checkpoint'], 'model_'+fname), map_location=device)
        model.load_state_dict(model_path['model_state_dict'])   
    ires = 0
    if args.resume == True:
        model_path = torch.load(os.path.join(cfg['checkpoint'], 'model_'+fname), map_location=device)
        model.load_state_dict(model_path['model_state_dict'])
        log = np.loadtxt(os.path.join(cfg['checkpoint'], 'log_'+fname+'.txt'), skiprows=1)
        ires = int(log[-1, 0]) + 1
    model.to(device)
        
    with open(args.checkpoint + "/config_"+fname+".txt", 'a') as f:
        f.write('Total model params.: {}'.format(sum(p.numel() for p in model.parameters())) + '\n')
    print('    Total model params.: {}'.format(sum(p.numel() for p in model.parameters())))
    
    # loss function and optimizer
    if args.loss_type == 'state' or args.loss_type == 'flux' or args.loss_type == 'comb':
        crion1 = nn.L1Loss(reduction='mean')
        crion2 = nn.MSELoss(reduction='mean')
        criterion = (crion1, crion2)
    elif args.loss_type == 'kld':
        criterion = nn.KLDivLoss(reduction='batchmean')
    elif args.loss_type == 'mixed':
        crion1 = nn.KLDivLoss(reduction='batchmean')
        crion2 = nn.MSELoss(reduction='mean')
        criterion = (crion1, crion2)
        

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,.99), weight_decay=args.weight_decay, amsgrad = True)
    # logger
    logger = Logger(os.path.join(args.checkpoint, 'log_'+fname+'.txt'), title = 'log', resume=args.resume)
    if ires == 0:
        logger.set_names(['Epoch', '        Learning Rate.', 'Train Loss.', 'Accu. u', '        Accu. theta_m', 
                          'Accu. R', '        Accu. theta_v' , '  Accu. E'])
    
    # load dataset
    data_load = scipy.io.loadmat(args.data_file1)
    tt = np.transpose(data_load.get('TT'), (1,0))
    coeff = data_load.get('coeff')
    ff1 = data_load.get('FF')
    um1 = data_load.get('MEAN')
    var1 = data_load.get('VARI')
    ene1 = data_load.get('ENE')
    theta1 = data_load.get('Theta')
    theta_v1 = data_load.get('THETA_V')
    data_load = scipy.io.loadmat(args.data_file2)
    ff2 = data_load.get('FF')
    um2 = data_load.get('MEAN')
    var2 = data_load.get('VARI')
    ene2 = data_load.get('ENE')
    theta2 = data_load.get('Theta')
    theta_v2 = data_load.get('THETA_V')
    
    # load data in the observed step
    args.norm = 200
    args.J = 40
    args.K = var1.shape[2]
    args.nm = int(args.J/2+1)
    args.coeff  = coeff
    # args.coeff1 = coeff1
    tt = tt[1:args.train_length*args.nskip+1:args.nskip]
    dt = tt[1]-tt[0]
    ff1 = ff1[:args.train_length*args.nskip:args.nskip]
    ff2 = ff2[:args.train_length*args.nskip:args.nskip]
    um1 = um1[1:args.train_length*args.nskip+1:args.nskip]
    um2 = um2[1:args.train_length*args.nskip+1:args.nskip]
    ene1 = ene1[1:args.train_length*args.nskip+1:args.nskip] / args.norm
    ene2 = ene2[1:args.train_length*args.nskip+1:args.nskip] / args.norm
    theta1 = (theta1[:args.train_length*args.nskip]+theta1[1:args.train_length*args.nskip+1]) / 2
    theta1 = np.mean(theta1.reshape((args.nskip,-1,args.K), order='F'), axis=0)
    theta2 = (theta2[:args.train_length*args.nskip]+theta2[1:args.train_length*args.nskip+1]) / 2
    theta2 = np.mean(theta2.reshape((args.nskip,-1,args.K), order='F'), axis=0)
    var1 = var1[:,1:args.train_length*args.nskip+1:args.nskip] / args.J
    var2 = var2[:,1:args.train_length*args.nskip+1:args.nskip] / args.J
    theta_v1 = theta_v1[:, :args.train_length*args.nskip]
    theta_v1 = np.mean(theta_v1.reshape((args.nm,args.nskip,-1,args.K), order='F'), axis=1)
    theta_v2 = theta_v2[:, :args.train_length*args.nskip]
    theta_v2 = np.mean(theta_v2.reshape((args.nm,args.nskip,-1,args.K), order='F'), axis=1)

    vareq = torch.from_numpy(var1[:,-1,10])
    nskip = 1 #args.input_length
    Nseq  = (args.train_length-args.input_length - args.npred+1) // nskip
    Ntraj = args.K
    Nsamp = Ntraj * Nseq
    if args.data_type == 'pertfu':
        Nsamp = 2*Nsamp
    args.iters = Nsamp//args.train_batch
    args.dt = dt[0]
    train_set   = torch.zeros(args.input_length + args.npred-1, Nsamp, 2*args.nm+3, dtype=torch.double)
    target_set  = torch.zeros(args.input_length + args.npred-1, Nsamp, 2*args.nm+3, dtype=torch.double)
    forcing_set = torch.zeros(args.input_length + args.npred,   Nsamp,              dtype=torch.double)
    for l in range(Ntraj):
        for i in range(Nseq):   
            if args.data_type == 'pertf' or args.data_type == 'pertfu':
                train_set[:, i+Nseq*l, 0] = torch.from_numpy(um1[i*nskip    :i*nskip + args.input_length + args.npred-1,l])
                train_set[:, i+Nseq*l, 1] = torch.from_numpy(ene1[i*nskip   :i*nskip + args.input_length + args.npred-1,l])
                for s in range(args.nm):
                    train_set[:, i+Nseq*l, s+2]     = torch.from_numpy(theta_v1[s, i*nskip:i*nskip + args.input_length + args.npred-1,l])
                    train_set[:, i+Nseq*l, s+2+args.nm] = torch.from_numpy(var1[s, i*nskip:i*nskip + args.input_length + args.npred-1,l])
                train_set[:, i+Nseq*l, 2*args.nm+2]      = torch.from_numpy(theta1[i*nskip:i*nskip + args.input_length + args.npred-1,l])
                target_set[:,i+Nseq*l, 0] = torch.from_numpy(um1[i*nskip+1    :i*nskip + args.input_length + args.npred,l])
                target_set[:,i+Nseq*l, 1] = torch.from_numpy(ene1[i*nskip+1   :i*nskip + args.input_length + args.npred,l])
                for s in range(args.nm):
                    target_set[:, i+Nseq*l, s+2]     = torch.from_numpy(theta_v1[s, i*nskip+1:i*nskip + args.input_length + args.npred,l])
                    target_set[:, i+Nseq*l, s+2+args.nm] = torch.from_numpy(var1[s, i*nskip+1:i*nskip + args.input_length + args.npred,l])
                target_set[:,i+Nseq*l, 2*args.nm+2]       = torch.from_numpy(theta1[i*nskip+1:i*nskip + args.input_length + args.npred,l])
                forcing_set[:,i+Nseq*l]    = torch.from_numpy(ff1[i*nskip :i*nskip + args.input_length + args.npred,l])
            elif args.data_type == 'pertu':
                train_set[:, i+Nseq*l, 0] = torch.from_numpy(um2[i*nskip    :i*nskip + args.input_length + args.npred-1,l])
                train_set[:, i+Nseq*l, 1] = torch.from_numpy(ene2[i*nskip   :i*nskip + args.input_length + args.npred-1,l])
                for s in range(args.nm):
                    train_set[:, i+Nseq*l, s+2]     = torch.from_numpy(theta_v2[s, i*nskip:i*nskip + args.input_length + args.npred-1,l])
                    train_set[:, i+Nseq*l, s+2+args.nm] = torch.from_numpy(var2[s, i*nskip:i*nskip + args.input_length + args.npred-1,l])
                train_set[:, i+Nseq*l, 2*args.nm+2]      = torch.from_numpy(theta2[i*nskip:i*nskip + args.input_length + args.npred-1,l])
                target_set[:,i+Nseq*l, 0]  = torch.from_numpy(um2[i*nskip+1   :i*nskip + args.input_length + args.npred,l])
                target_set[:,i+Nseq*l, 1] = torch.from_numpy(ene2[i*nskip+1   :i*nskip + args.input_length + args.npred,l])
                for s in range(args.nm):
                    target_set[:, i+Nseq*l, s+2]     = torch.from_numpy(theta_v2[s, i*nskip+1:i*nskip + args.input_length + args.npred,l])
                    target_set[:, i+Nseq*l, s+2+args.nm] = torch.from_numpy(var2[s, i*nskip+1:i*nskip + args.input_length + args.npred,l])
                target_set[:,i+Nseq*l, 2*args.nm+2]       = torch.from_numpy(theta2[i*nskip+1:i*nskip + args.input_length + args.npred,l])
                forcing_set[:,i+Nseq*l]    = torch.from_numpy(ff2[i*nskip :i*nskip + args.input_length + args.npred,l])
                
            if args.data_type == 'pertfu':
                train_set[:, i+Nseq*(l+Ntraj), 0] = torch.from_numpy(um2[i*nskip    :i*nskip + args.input_length + args.npred-1,l])
                train_set[:, i+Nseq*(l+Ntraj), 1] = torch.from_numpy(ene2[i*nskip   :i*nskip + args.input_length + args.npred-1,l])
                for s in range(args.nm):
                    train_set[:,i+Nseq*(l+Ntraj), s+2]     = torch.from_numpy(theta_v2[s, i*nskip:i*nskip + args.input_length + args.npred-1,l])
                    train_set[:,i+Nseq*(l+Ntraj), s+2+args.nm] = torch.from_numpy(var2[s, i*nskip:i*nskip + args.input_length + args.npred-1,l])
                train_set[:,i+Nseq*(l+Ntraj), 2*args.nm+2]      = torch.from_numpy(theta2[i*nskip:i*nskip + args.input_length + args.npred-1,l])
                target_set[:,i+Nseq*(l+Ntraj), 0]  = torch.from_numpy(um2[i*nskip+1   :i*nskip + args.input_length + args.npred,l])
                target_set[:,i+Nseq*(l+Ntraj), 1] = torch.from_numpy(ene2[i*nskip+1   :i*nskip + args.input_length + args.npred,l])
                for s in range(args.nm):
                    target_set[:,i+Nseq*(l+Ntraj), s+2]     = torch.from_numpy(theta_v2[s, i*nskip+1:i*nskip + args.input_length + args.npred,l])
                    target_set[:,i+Nseq*(l+Ntraj), s+2+args.nm] = torch.from_numpy(var2[s, i*nskip+1:i*nskip + args.input_length + args.npred,l])
                target_set[:,i+Nseq*(l+Ntraj), 2*args.nm+2]    = torch.from_numpy(theta2[i*nskip+1:i*nskip + args.input_length + args.npred,l])
                forcing_set[:,i+Nseq*(l+Ntraj)]    = torch.from_numpy(ff2[i*nskip :i*nskip + args.input_length + args.npred,l])
        
    train_loader = (train_set, target_set, forcing_set, vareq)
    del(data_load,ff1,ff2,um1,um2,var1,var2,ene1,ene2,theta1,theta2,theta_v1,theta_v2)
    

    # training performance measure
    epoch_loss = np.zeros((args.epoch, 2))
    epoch_accu = np.zeros((args.epoch, 2))
    epoch_accE = np.zeros((args.epoch, 2))
    epoch_accr  = np.zeros((args.epoch,args.nm, 2))
    epoch_accth = np.zeros((args.epoch,args.nm+1, 2))
    for epoch in range(ires, args.epoch):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [{} | {}] LR: {:.8f} loss: {}'.format(epoch + 1, cfg['epoch'], cfg['lr'], cfg['loss_type']))
        train_loss,vloss, train_accu,vaccu, train_accE,vaccE, train_accr,vaccr, train_accth,vaccth, \
        train_accthv,vaccthv, pred, gt = train(train_loader, model, criterion, optimizer, device)

        # save accuracy
        epoch_loss[epoch,0]  = train_loss
        epoch_accu[epoch,0]  = train_accu
        epoch_accE[epoch,0]  = train_accE
        epoch_accr[epoch,:,0] = train_accr
        epoch_accth[epoch,0,0] = train_accth
        epoch_accth[epoch,1:,0] = train_accthv
        epoch_loss[epoch,1]  = vloss
        epoch_accu[epoch,1]  = vaccu
        epoch_accE[epoch,1]  = vaccE
        epoch_accr[epoch,:,1] = vaccr
        epoch_accth[epoch,0,1] = vaccth
        epoch_accth[epoch,1:,1] = vaccthv
        
        # append logger file
        logger.append([epoch, cfg['lr'], train_loss, train_accu,train_accth, train_accr.sum(), train_accthv.sum(),train_accE])
        filepath = os.path.join(cfg['checkpoint'], 'model_' + fname)
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),}, filepath)
        
    datapath = os.path.join(cfg['checkpoint'], 'train_' + fname)
    np.savez(datapath, tt = tt, epoch_loss = epoch_loss, epoch_accu = epoch_accu, epoch_accr = epoch_accr,
             epoch_accth = epoch_accth, epoch_accE = epoch_accE, pred = pred, gt = gt) 
    
    # evaluating model in prediction data set
    if valid:
        # load evaluation dataset
        data_load = scipy.io.loadmat(args.pred_file1)
        tt = np.transpose(data_load.get('TT'), (1,0))
        ff1 = np.transpose(data_load.get('FF'), (1,0))
        um1 = np.transpose(data_load.get('MEAN'), (1,0))
        var1 = np.transpose(data_load.get('VARI'), (1,0))
        ene1 = np.transpose(data_load.get('ENE'), (1,0))
        theta1 = np.transpose(data_load.get('Theta'), (1,0))
        theta_v1 = np.transpose(data_load.get('THETA_V'), (1,0))
        data_load = scipy.io.loadmat(args.pred_file2)
        ff2 = np.transpose(data_load.get('FF'), (1,0))
        um2 = np.transpose(data_load.get('MEAN'), (1,0))
        var2 = np.transpose(data_load.get('VARI'), (1,0))
        ene2 = np.transpose(data_load.get('ENE'), (1,0))
        theta2 = np.transpose(data_load.get('Theta'), (1,0))
        theta_v2 = np.transpose(data_load.get('THETA_V'), (1,0))
        data_load = scipy.io.loadmat(args.pred_file3)
        ff3 = np.transpose(data_load.get('FF'), (1,0))
        um3 = np.transpose(data_load.get('MEAN'), (1,0))
        var3 = np.transpose(data_load.get('VARI'), (1,0))
        ene3 = np.transpose(data_load.get('ENE'), (1,0))
        theta3 = np.transpose(data_load.get('Theta'), (1,0))
        theta_v3 = np.transpose(data_load.get('THETA_V'), (1,0))
        data_load = scipy.io.loadmat(args.pred_file4)
        ff4 = np.transpose(data_load.get('FF'), (1,0))
        um4 = np.transpose(data_load.get('MEAN'), (1,0))
        var4 = np.transpose(data_load.get('VARI'), (1,0))
        ene4 = np.transpose(data_load.get('ENE'), (1,0))
        theta4 = np.transpose(data_load.get('Theta'), (1,0))
        theta_v4 = np.transpose(data_load.get('THETA_V'), (1,0))
    

        tt = tt[:args.pred_length*args.nskip+1:args.nskip]
        dt = tt[1]-tt[0]
        ff1  = ff1[:args.pred_length*args.nskip+1:args.nskip]
        ff2  = ff2[:args.pred_length*args.nskip+1:args.nskip]
        ff3  = ff3[:args.pred_length*args.nskip+1:args.nskip]
        ff4  = ff4[:args.pred_length*args.nskip+1:args.nskip]
        um1  = um1[:args.pred_length*args.nskip+1:args.nskip]
        um2  = um2[:args.pred_length*args.nskip+1:args.nskip]
        um3  = um3[:args.pred_length*args.nskip+1:args.nskip]
        um4  = um4[:args.pred_length*args.nskip+1:args.nskip]
        var1 = var1[:args.pred_length*args.nskip+1:args.nskip] / args.J
        var2 = var2[:args.pred_length*args.nskip+1:args.nskip] / args.J
        var3 = var3[:args.pred_length*args.nskip+1:args.nskip] / args.J
        var4 = var4[:args.pred_length*args.nskip+1:args.nskip] / args.J
        ene1 = ene1[:args.pred_length*args.nskip+1:args.nskip] / args.norm
        ene2 = ene2[:args.pred_length*args.nskip+1:args.nskip] / args.norm
        ene3 = ene3[:args.pred_length*args.nskip+1:args.nskip] / args.norm
        ene4 = ene4[:args.pred_length*args.nskip+1:args.nskip] / args.norm
        theta1 = (theta1[:args.pred_length*args.nskip]+theta1[1:args.pred_length*args.nskip+1]) / 2
        theta1 = np.mean(theta1.reshape((args.nskip,-1), order='F'), axis=0)
        theta2 = (theta2[:args.pred_length*args.nskip]+theta2[1:args.pred_length*args.nskip+1]) / 2
        theta2 = np.mean(theta2.reshape((args.nskip,-1), order='F'), axis=0)
        theta3 = (theta3[:args.pred_length*args.nskip]+theta3[1:args.pred_length*args.nskip+1]) / 2
        theta3 = np.mean(theta3.reshape((args.nskip,-1), order='F'), axis=0)
        theta4 = (theta4[:args.pred_length*args.nskip]+theta4[1:args.pred_length*args.nskip+1]) / 2
        theta4 = np.mean(theta4.reshape((args.nskip,-1), order='F'), axis=0)
        theta_v1 = theta_v1[:args.pred_length*args.nskip]
        theta_v1 = np.mean(theta_v1.reshape((args.nskip,-1,args.nm), order='F'), axis=0)
        theta_v2 = theta_v2[:args.pred_length*args.nskip]
        theta_v2 = np.mean(theta_v2.reshape((args.nskip,-1,args.nm), order='F'), axis=0)
        theta_v3 = theta_v3[:args.pred_length*args.nskip]
        theta_v3 = np.mean(theta_v3.reshape((args.nskip,-1,args.nm), order='F'), axis=0)
        theta_v4 = theta_v4[:args.pred_length*args.nskip]
        theta_v4 = np.mean(theta_v4.reshape((args.nskip,-1,args.nm), order='F'), axis=0)
        
        npred = args.pred_length-args.input_length
        traj_set = torch.zeros(args.pred_length+1, 4, 2*args.nm+4, dtype=torch.double)
        traj_set[:,0, 0] = torch.from_numpy(um1)[:,0]
        traj_set[:,0, 1] = torch.from_numpy(ene1)[:,0]
        traj_set[1:,0, 2:2+args.nm] = torch.from_numpy(theta_v1)
        traj_set[:,0, 2+args.nm:2*args.nm+2] = torch.from_numpy(var1)
        traj_set[1:,0, 2*args.nm+2] = torch.from_numpy(theta1)
        traj_set[:,0, 2*args.nm+3] = torch.from_numpy(ff1)[:,0]
        traj_set[:,1, 0] = torch.from_numpy(um2)[:,0]
        traj_set[:,1, 1] = torch.from_numpy(ene2)[:,0]
        traj_set[1:,1, 2:2+args.nm] = torch.from_numpy(theta_v2)
        traj_set[:,1, 2+args.nm:2*args.nm+2] = torch.from_numpy(var2)
        traj_set[1:,1, 2*args.nm+2] = torch.from_numpy(theta2)
        traj_set[:,1, 2*args.nm+3] = torch.from_numpy(ff2)[:,0]
        traj_set[:,2, 0] = torch.from_numpy(um3)[:,0]
        traj_set[:,2, 1] = torch.from_numpy(ene3)[:,0]
        traj_set[1:,2, 2:2+args.nm] = torch.from_numpy(theta_v3)
        traj_set[:,2, 2+args.nm:2*args.nm+2] = torch.from_numpy(var3)
        traj_set[1:,2, 2*args.nm+2] = torch.from_numpy(theta3)
        traj_set[:,2, 2*args.nm+3] = torch.from_numpy(ff3)[:,0]
        traj_set[:,3, 0] = torch.from_numpy(um4)[:,0]
        traj_set[:,3, 1] = torch.from_numpy(ene4)[:,0]
        traj_set[1:,3, 2:2+args.nm] = torch.from_numpy(theta_v4)
        traj_set[:,3, 2+args.nm:2*args.nm+2] = torch.from_numpy(var4)
        traj_set[1:,3, 2*args.nm+2] = torch.from_numpy(theta4)
        traj_set[:,3, 2*args.nm+3] = torch.from_numpy(ff4)[:,0]
        
        input_set = (traj_set, vareq)
        logger.file.write('\n')
        logger.set_names(['Model eval.', 'total', '        error u', '        error theta_m',
                          ' error r', ' error theta_v'])
        valid_pred, valid_err = prediction(input_set, npred, model, logger, device)
        
        datapath = os.path.join(cfg['checkpoint'], 'pred_' + fname)
        np.savez(datapath, tt = tt, pred = valid_pred[:,:,:,0], gt = valid_pred[:,:,:,1], valid_err = valid_err)

    logger.close()
    
def prediction(input_set, npred, model, logger, device):
    with torch.no_grad():
        model.eval()
    l96_inte = l96_sys.l96_2mom(dt = args.dt, coeff=args.coeff, nm = args.norm, device=device)
 
    valid_pred = np.zeros((npred, 4,2*args.nm+3, 2))
    valid_err  = np.zeros((npred, 2*args.nm+3))

    traj_set, vareq = input_set
    vareq = vareq.to(device)
    istate = traj_set[1:args.input_length+1, :,:args.nm+2].clone().to(device)
    r_in   = traj_set[1:args.input_length+1, :,args.nm+2:2*args.nm+2].clone().to(device)
    hidden = ()
    with torch.no_grad():
        for istep in range(npred):
            # target set data
            target  = traj_set[(istep+2): args.input_length + (istep+2), :,:2*args.nm+3]
            forcing = traj_set[(istep+1): args.input_length + (istep+2), :,2*args.nm+3].to(device, non_blocking=True)

            # run model in one forward iteration
            qm_out, hidden = model(istate, hidden, device=device)
            dm_out = -torch.min(qm_out, torch.tensor([0.], dtype=torch.double,device=device)) / vareq.view(1,1,-1).expand_as(qm_out)
            sm_out = torch.max(qm_out, torch.tensor([0.],  dtype=torch.double,device=device))
            thv_out = -dm_out*r_in + sm_out
            u_out,th_out, E_out,r_out = l96_inte.var_mean_ene(istate,r_in, thv_out, forcing)
            
            
            istate[:-1,:,:] = istate[1:,:,:].clone()
            istate[-1,:,0]  = u_out[-1]
            istate[-1,:,1]  = E_out[-1]
            istate[-1,:,2:] = thv_out[-1]
            r_in[:-1,:,:]   = r_in[1:,:,:].clone()
            r_in[-1,:,:]    = r_out[-1]

            predu  = u_out.data.cpu().numpy()[-1]
            predE  = E_out.data.cpu().numpy()[-1]
            predr  = r_out.data.cpu().numpy()[-1]
            predth = th_out.data.cpu().numpy()[-1]
            predthv = thv_out.data.cpu().numpy()[-1]
            targ  = target.data.cpu().numpy()[-1]
            valid_pred[istep, :,0, 0] = predu
            valid_pred[istep, :,1, 0] = predE
            valid_pred[istep, :,2:2+args.nm, 0] = predthv
            valid_pred[istep, :,2+args.nm:2+2*args.nm, 0] = predr
            valid_pred[istep, :,2+2*args.nm, 0] = predth
            valid_pred[istep, :,:, 1] = targ
            
            valid_err[istep,0] = np.sqrt( np.square(predu  - targ[:,0]).mean())
            valid_err[istep,1] = np.sqrt( np.square(predE  - targ[:,1]).mean())
            valid_err[istep,2:2+args.nm] = np.sqrt( np.square(predthv  - targ[:,2:2+args.nm]).mean(0))
            valid_err[istep,2+args.nm:2+2*args.nm] = np.sqrt( np.square(predr  - targ[:,2+args.nm:2+2*args.nm]).mean(0))
            valid_err[istep,2+2*args.nm] = np.sqrt( np.square(predth - targ[:,2+2*args.nm]).mean())


            print('step {}: error_u = {:.6f} error_thr = {:.6f}  error_r = {:.6f}'.format(istep, valid_err[istep,0], 
                  valid_err[istep,2:2+args.nm].sum(), valid_err[istep,2+args.nm:2+2*args.nm].sum()))
            logger.append([istep, valid_err[istep,:].sum(), valid_err[istep,0],valid_err[istep,2+2*args.nm], 
                           valid_err[istep,2+args.nm:2+2*args.nm].sum(), valid_err[istep,2:2+args.nm].sum()])
        
    return valid_pred, valid_err
    
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    l96_inte = l96_sys.l96_2mom(dt=args.dt, coeff=args.coeff, nm=args.norm, device=device)
    
    batch_time = AverageMeter()
    losses     = AverageMeter()
    accsu      = AverageMeter()
    accsE      = AverageMeter()
    accsr      = AverageMeter()
    accsth     = AverageMeter()
    accsthv    = AverageMeter()
    end = time.time()
    
    input_full, target_full, forcing_full, vareq = train_loader
    vareq = vareq.to(device)
    dsize = args.train_batch*args.iters
    s_idx = random.sample(range(0,input_full.size(1)), dsize)
    input_iter   = input_full[:, s_idx,:].pin_memory()
    target_iter  = target_full[:,s_idx,:].pin_memory()
    forcing_iter = forcing_full[:, s_idx].pin_memory()
    
    for ib in range(0, args.iters):
        inputs  = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, :].to(device, non_blocking=True)
        targets = target_iter[:,ib*args.train_batch:(ib+1)*args.train_batch, :].to(device, non_blocking=True)
        forcing = forcing_iter[:,ib*args.train_batch:(ib+1)*args.train_batch].to(device, non_blocking=True)
        
        optimizer.zero_grad()  # zero the gradient buffers
        # iteration the model in npred steps
        hidden = ()
        istate = inputs[:args.input_length,:,:args.nm+2]
        r_in   = inputs[:args.input_length,:,args.nm+2:2*args.nm+2]
        
        pred = torch.empty(args.input_length+args.npred, args.train_batch, 2*args.nm+3, dtype=torch.double, device=device)
        pred[:args.input_length,:,:] = inputs[:args.input_length,:,:].clone()
        loss = 0
        for ip in range(args.npred):
            qm_out, hidden = model(istate, hidden, device=device)
            dm_out = -torch.min(qm_out, torch.tensor([0.], dtype=torch.double,device=device)) / vareq.view(1,1,-1).expand_as(qm_out)
            sm_out =  torch.max(qm_out, torch.tensor([0.], dtype=torch.double,device=device))
            thv_out = -dm_out*r_in + sm_out
            u_out,th_out, E_out,r_out = l96_inte.var_mean_ene(istate,r_in, thv_out, forcing[ip:ip+args.input_length+1])

            
            pred[args.input_length+ip,:, 0]                     = u_out[-1]
            pred[args.input_length+ip,:, 1]                     = E_out[-1]
            pred[args.input_length+ip,:, 2:2+args.nm]           = thv_out[-1]
            pred[args.input_length+ip,:, 2+args.nm:2+2*args.nm] = r_out[-1]
            pred[args.input_length+ip,:, 2+2*args.nm]           = th_out[-1]
            
            if ip < args.npred-1:
                istate = torch.empty_like(istate)
                r_in = torch.empty_like(r_in)
                # update with final model output
                # istate[:,:,:] = pred[ip+1:args.input_length+ip+1,:,:2+args.nm]
                # r_in[:,:,:] = pred[ip+1:args.input_length+ip+1,:,2+args.nm:2+2*args.nm]
                # update with full model output
                istate[:,:,0]  = u_out.clone()
                istate[:,:,1]  = E_out.clone()
                istate[:,:,2:] = thv_out.clone()
                r_in[:,:,:] = r_out #.clone()

            output = torch.transpose(torch.cat([u_out[:,:,None],E_out[:,:,None],thv_out,r_out,th_out[:,:,None]],2), 0,1)
            target = torch.transpose(targets[ip:args.input_length+ip,:,:], 0,1)
            if args.loss_type == 'state':
                crion1, crion2 = criterion
                out1 = output[:, -args.nloss:, :2]
                tag1 = target[:, -args.nloss:, :2]
                out2 = output[:, -args.nloss:, args.nm+2:2*args.nm+2]
                tag2 = target[:, -args.nloss:, args.nm+2:2*args.nm+2]
                loss += 1.*crion2(out1, tag1) + 1.*crion1(out2, tag2)
            elif args.loss_type == 'flux':
                crion1, crion2 = criterion
                out1 = output[:, -args.nloss:, 2:args.nm+2]
                tag1 = target[:, -args.nloss:, 2:args.nm+2]
                out2 = output[:, -args.nloss:, 2*args.nm+2]
                tag2 = target[:, -args.nloss:, 2*args.nm+2]
                loss += crion1(out1, tag1) + 1.*crion2(out2, tag2)
            elif args.loss_type == 'comb':
                crion1, crion2 = criterion
                out1 = output[:, -args.nloss:, :2]
                tag1 = target[:, -args.nloss:, :2]
                out2 = output[:, -args.nloss:, args.nm+2:2*args.nm+2]
                tag2 = target[:, -args.nloss:, args.nm+2:2*args.nm+2]
                out3 = output[:, -args.nloss:, 2:args.nm+2]
                tag3 = target[:, -args.nloss:, 2:args.nm+2]
                out4 = output[:, -args.nloss:, 2*args.nm+2]
                tag4 = target[:, -args.nloss:, 2*args.nm+2]
                loss += 1.*crion2(out1, tag1) + 1.*crion1(out2, tag2)
                loss += 1.*crion1(out3, tag3) + 1.*crion2(out4, tag4)
                
        
        loss.backward()
        optimizer.step()
        
        # get trained output
        losses.update(loss.item() )
        pred_out = pred[args.input_length:]
        gt_out   = targets[args.input_length-1:]
        accsu.update( ((pred_out[:,:,0]-gt_out[:,:,0]).square().mean()).item() ) # / gt_out[:,:,0].square().sum()).item() )
        accsE.update( ((pred_out[:,:,1]-gt_out[:,:,1]).square().mean()).item() )
        accsthv.update( ((pred_out[:,:,2:2+args.nm]-gt_out[:,:,2:2+args.nm]).square().mean(1).mean(0) ).data.cpu().numpy() )
        accsr.update( ((pred_out[:,:,2+args.nm:2+2*args.nm]-gt_out[:,:,2+args.nm:2+2*args.nm]).square().mean(1).mean(0) ).data.cpu().numpy() ) # / gt_out[:,:,1].square().sum()).item() )
        accsth.update(((pred_out[:,:,2+2*args.nm]-gt_out[:,:,2+2*args.nm]).square().mean()).item() ) # / gt_out[:,:,2].square().sum()).item() )
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        
        #LR = optimizer.param_groups[0]['lr']
        suffix = 'Iter{iter}: Loss = {loss:.5e} u = {accu:.5e} R = {accr:.5e} th = {accth:.5e}  thv = {accthv:.5e} E = {accE:.5e}  run_time = {bt:.2f}'.format(
                  iter = ib, loss = losses.val, accu=accsu.val,accr=accsr.val.sum(), accth=accsth.val,accthv=accsthv.val.sum(), accE=accsE.val, bt = batch_time.sum)
        if ib % 20 == 0:
            print(suffix)
            
    # get trained output
    pred = pred_out.data.cpu().numpy()
    gt   = gt_out.data.cpu().numpy()

    return losses.avg,losses.var, accsu.avg,accsu.var,accsE.avg,accsE.var, accsr.avg,accsr.var, accsth.avg,accsth.var,accsthv.avg,accsthv.var, pred, gt
    
def adjust_learning_rate(optimizer, epoch):
    global cfg
    if epoch in cfg['schedule']:
        cfg['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['lr']

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    gflags.DEFINE_boolean('pretrained', False, 'Use pretrained model')
    gflags.DEFINE_boolean('eval', True, 'Run tests with the network')
    
    main(pretrained = gflags.FLAGS.pretrained, valid = gflags.FLAGS.eval)
