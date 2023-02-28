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
parser.add_argument('--train_length', default=300, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--input_length', default=100, type=int, metavar='L',
                    help='model input state size')
parser.add_argument('--iters', default=100, type=int, metavar='I',
                    help='number of iterations for each epoch')
parser.add_argument('--train-batch', default=500, type=int, metavar='B',
                    help='each training batch size')
parser.add_argument('--nskip', default=10, type=int, metavar='nk',
                    help='time step skip in the loaded raw data, dt=1e-2')
parser.add_argument('--npred', default=20, type=int, metavar='Np',
                    help='number of iterations to measure in the loss func.')
parser.add_argument('--nopt', default=20, type=int, metavar='Nopt',
                    help='number of iterations before updating the loss func.')
# (train-batch * iters = train_length - input_length - npred+1)
parser.add_argument('--indmk', default=[0,1,2,3, 21,22,23], type=int, metavar='k1',
                    help='resolved wavenumbers in red. model, must include real and imaginary parts')
parser.add_argument('--indrk', default=[6,7,8,9,10,11,12, 26,27,28,29,30,31,32], type=int, metavar='k1',
                    help='resolved wavenumbers in red. model, must include real and imaginary parts')


# model parameters
parser.add_argument('--epoch', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nhidrv', default=500, type=int, metavar='nh',
                    help='hidden layer size in the network cell')
parser.add_argument('--nhidmk', default=500, type=int, metavar='nhm',
                    help='hidden layer size in the network cell for mean uk')
parser.add_argument('--nhidm0', default=200, type=int, metavar='nhm',
                    help='hidden layer size in the network cell for mean u0')
parser.add_argument('--nloss', default=100, type=int, metavar='nv',
                    help='number of steps to measure in the loss function')
# optimization parameters
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25, 50, 75],
                    help='Decrease learning rate at these epochs. [25, 50,75]')
parser.add_argument('--gamma', type=float, default=0.5, 
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--loss-type', '--lt', default='kld', type=str, metavar='LT',
                    help='choices of loss functions (state,flux,comb, kld,mixed)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/train_stat_inhomo_F8pertk', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/pertk/l96_stat2_nt1_inhomo_F8df15dd5', type=str, metavar='DATA_PATH',
                    help='path to train data set pert forcing')
parser.add_argument('--samp-file', default='data/pertk/l96_samp2_nt1_inhomo_F8df15dd5', type=str, metavar='DATA_PATH',
                    help='path to train data set pert forcing')
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
fname = 'stat0_pertk1_nb{}op20_nredm{}r{}_adamlr01_ls1stg0hv{}hm{}h{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.train_batch, 
            len(args.indmk),len(args.indrk), args.nhidrv,args.nhidmk,args.nhidm0, args.nloss, 
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
with open(args.checkpoint + "/config_"+fname+".txt", 'w') as f:
    for (k, v) in args._get_kwargs():
        f.write(k + ' : ' + str(v) + '\n')
    f.write('\n')

def main(pretrained = False, valid = False):
    args.nred = len(args.indmk)
    args.nmnk = len(args.indmk)-1
    args.nmp  = int((args.nred+1)/2)
    args.indm = args.indmk[:args.nmp]
    args.nvar = int(len(args.indrk)/2)
    args.indr = args.indrk[:args.nvar]
    # models for unresolved processes
    model_m0 = models.LSTMresi(input_size = args.nred+args.nvar+1,         hidden_size = args.nhidm0, output_size = 1, 
                                nlayers = 1, nstages = 0).double()
    model_mk = models.LSTMresi(input_size = args.nred+args.nvar+args.nmnk, hidden_size = args.nhidmk, output_size = args.nmnk, 
                                nlayers = 1, nstages = 0).double()
    model_rv = models.LSTMresi(input_size = args.nred+2*args.nvar,         hidden_size = args.nhidrv, output_size = args.nvar, 
                                nlayers = 1, nstages = 0).double()
    # load model on GPU
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('This code is run by {}: {} GPU(s)'.format(dev, torch.cuda.device_count()))
    # if  torch.cuda.device_count() > 1:
    #     model_m = nn.DataParallel(model_m).to(dev1)
    #     model_v = nn.DataParallel(model_v).to(dev2)
    # cudnn.benchmark = True
    # cudnn.enabled = True
    
    if pretrained:
        # load the pretrained model
        model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'modelm0_'+fname), map_location=dev)
        model_m0.load_state_dict(model_path1['model_state_dict'])
        model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'modelmk_'+fname), map_location=dev)
        model_mk.load_state_dict(model_path2['model_state_dict'])
        model_path3 = torch.load(os.path.join(cfg['checkpoint'], 'modelrv_'+fname), map_location=dev)
        model_rv.load_state_dict(model_path3['model_state_dict'])
    ires = 0
    if args.resume == True:
        model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'modelm0_'+fname), map_location=dev)
        model_m0.load_state_dict(model_path1['model_state_dict'])
        model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'modelmk_'+fname), map_location=dev)
        model_mk.load_state_dict(model_path2['model_state_dict'])
        model_path3 = torch.load(os.path.join(cfg['checkpoint'], 'modelrv_'+fname), map_location=dev)
        model_rv.load_state_dict(model_path3['model_state_dict'])
        log = np.loadtxt(os.path.join(cfg['checkpoint'], 'log_'+fname+'.txt'), skiprows=1)
        ires = int(log[-1, 0]) + 1
    model_m0.to(dev)
    model_mk.to(dev)
    model_rv.to(dev)
    model = (model_m0, model_mk, model_rv)

    with open(args.checkpoint + "/config_"+fname+".txt", 'a') as f:
        f.write('Total model params. for mean u0: {}'.format(sum(p.numel() for p in model_m0.parameters())) + '\n')
        f.write('Total model params. for mean uk: {}'.format(sum(p.numel() for p in model_mk.parameters())) + '\n')
        f.write('Total model params. for variance: {}'.format(sum(p.numel() for p in model_rv.parameters())) + '\n')
    print('    Total mean0 model params.: {}'.format(sum(p.numel() for p in model_m0.parameters())))
    print('    Total meank model params.: {}'.format(sum(p.numel() for p in model_mk.parameters())))
    print('    Total vari model params.: {}'.format(sum(p.numel() for p in model_rv.parameters())))

    
    # loss function and optimizer
    crion1 = nn.L1Loss(reduction='mean')
    crion2 = nn.MSELoss(reduction='mean')
    criterion = (crion1, crion2)
    
    optim_m0 = optim.Adam(model_m0.parameters(), lr=args.lr, betas=(0.9,.99), weight_decay=args.weight_decay, amsgrad = True)
    optim_mk = optim.Adam(model_mk.parameters(), lr=args.lr, betas=(0.9,.99), weight_decay=args.weight_decay, amsgrad = True)
    optim_rv = optim.Adam(model_rv.parameters(), lr=args.lr, betas=(0.9,.99), weight_decay=args.weight_decay, amsgrad = True)
    optimizer = (optim_m0, optim_mk, optim_rv)

    # logger
    logger = Logger(os.path.join(args.checkpoint, 'log_'+fname+'.txt'), title = 'log', resume=args.resume)
    if ires == 0:
        logger.set_names(['Epoch', '        Learning Rate.', 'Train Loss.', 'Accu. u0', 'Accu. uk', 'Accu. R','        Accu. theta_v', 'Accu. theta_m0', 'Accu. theta_mk'])
    
    # load dataset
    data_load = scipy.io.loadmat(args.data_file)
    tt = np.transpose(data_load.get('TT'), (1,0))
    coeff_c = data_load.get('coeff_c')[:,0]
    coeff   = data_load.get('coeff')
    fk      = data_load.get('fk')
    dk      = data_load.get('dk')
    dki     = data_load.get('dk1')
    Dk = np.concatenate((dk,dki[:,1:-1]), axis=1)
    Dkm = Dk[args.indm]
    Dkm = Dkm[:,args.indmk]
    Dkv = Dk[args.indr]
    Dkv = Dkv[:,args.indrk]
    
    um      = data_load.get('MEAN')
    var     = data_load.get('VARI')
    cov     = data_load.get('COV_SP')
    # diff_m   = data_load.get('DIFF_M')
    # diff_v = data_load.get('DIFF_V')
    # theta   = data_load.get('THETA_M')
    # theta_v = data_load.get('THETA_V')
    # theta_z = np.transpose(data_load.get('THETA_Z'), (0,2,1,3,4))
    # load samples
    data_load = scipy.io.loadmat(args.samp_file)
    zt = np.transpose(data_load.get('Ztraj'), (0,2,1,3,4))

    
    # load data in the observed step
    # args.norm = 1
    args.J = 40
    args.K = um.shape[2]
    args.L = um.shape[3]
    args.nm = int(args.J/2+1)
    args.coeff_cr  = coeff_c.real
    args.coeff_ci  = coeff_c.imag
    args.coeff = coeff
    # args.Dk_r = Dk.real
    # args.Dk_i = Dk.imag
    # args.dkv = dk[args.indr,args.indr,:].real
    nsamp = args.train_batch #zt.shape[2]
    args.nsamp = nsamp
    
    ddt = tt[1]-tt[0]
    tt = tt[args.nskip:args.train_length*args.nskip+args.nskip:args.nskip]
    dt = tt[1]-tt[0]
    vareq = torch.from_numpy(var[:,0,0,0]) / args.J
    # combined damping & forcing effects
    Fk = np.repeat(fk[args.indm, :, np.newaxis], args.L, axis=2)
    DampU = np.zeros((args.nmp,  um.shape[1], args.K,args.L), dtype=complex)
    DampR = np.zeros((args.nvar,var.shape[1], args.K,args.L))
    for k in range(args.K):
        for l in range(args.L):
            for i in range(um.shape[1]):
                DampU[:,i,k,l]  = np.matmul(Dkm[:,:args.nmp,l], um[args.indm,i,k,l])
                DampU[:,i,k,l] += np.matmul(Dkm[:,args.nmp:,l], np.conjugate(um[args.indm[1:],i,k,l]))
                
                cov_red = cov[args.indrk,:,i,k,l] / args.J
                cov_red = cov_red[:,args.indr]
                DampR[:,i,k,l]  = 2*np.matmul(Dkv[:,:,l], cov_red).diagonal().real 
    DampU = (DampU[:,:args.train_length*args.nskip]+DampU[:,1:args.train_length*args.nskip+1]) / 2
    DampU = np.mean(DampU.reshape((args.nmp,args.nskip,-1,args.K,args.L), order='F'), axis=1)
    DampR = (DampR[:,:args.train_length*args.nskip]+DampR[:,1:args.train_length*args.nskip+1]) / 2
    DampR = np.mean(DampR.reshape((args.nvar,args.nskip,-1,args.K,args.L), order='F'), axis=1)
    # compute the reduced order mean feedback
    diff_m = (um[args.indm,1:] - um[args.indm,:-1]) / ddt 
    diff_m = np.mean(diff_m.reshape((args.nmp,args.nskip,-1,args.K,args.L), order='F'), axis=1)
    diff_m = diff_m[:,:args.train_length]
    ene1 = (coeff[args.indr,:,np.newaxis,np.newaxis]*var[args.indr,:,:,:]).sum(0) / args.J
    ene2 = (coeff[args.indm,:,np.newaxis,np.newaxis]*(np.absolute(um[args.indm,:,:,:])**2) ).sum(0)
    ene = ene1 + ene2
    ene = (ene[:args.train_length*args.nskip]+ene[1:args.train_length*args.nskip+1]) / 2
    ene = np.mean(ene.reshape((args.nskip,-1,args.K,args.L), order='F'), axis=0)
    theta_m0 = diff_m[0].real - ene - Fk[0,np.newaxis,:,:].real + DampU[0].real
    mcrx = coeff_c[args.indm[1:],np.newaxis,np.newaxis,np.newaxis]*um[0,:,:,:]*um[args.indm[1:],:,:,:]
    mcrx = (mcrx[:,:args.train_length*args.nskip]+mcrx[:,1:args.train_length*args.nskip+1]) / 2
    mcrx = np.mean(mcrx.reshape((args.nmp-1,args.nskip,-1,args.K,args.L), order='F'), axis=1)
    theta_mk = diff_m[1:] + mcrx - Fk[1:,np.newaxis,:,:] + DampU[1:]
    theta_mr = theta_mk.real
    theta_mi = theta_mk.imag
    # compute tthe reduced order variance feedback
    diff_v = (var[args.indr,1:] - var[args.indr,:-1]) / ddt / args.J
    diff_v = np.mean(diff_v.reshape((args.nvar,args.nskip,-1,args.K,args.L), order='F'), axis=1)
    diff_v = diff_v[:,:args.train_length]
    vcrx = 2*(coeff_c[args.indr,np.newaxis,np.newaxis,np.newaxis].real)*(um[0:1,:,:,:].real)*var[args.indr,:,:,:] / args.J
    vcrx = (vcrx[:,:args.train_length*args.nskip]+vcrx[:,1:args.train_length*args.nskip+1]) / 2
    vcrx = np.mean(vcrx.reshape((args.nvar,args.nskip,-1,args.K,args.L), order='F'), axis=1)
    theta_v = diff_v + vcrx + DampR
    
    um  =  um[args.indm,args.nskip:args.train_length*args.nskip+args.nskip:args.nskip]
    um_r = um.real
    um_i = um.imag
    var = var[args.indr,args.nskip:args.train_length*args.nskip+args.nskip:args.nskip] / args.J
    # cov = cov[args.indk,args.indk,args.nskip:args.train_length*args.nskip+args.nskip:args.nskip]
    zt  =  zt[args.indr,:args.train_length]
    zt_r = zt.real
    zt_i = zt.imag

    indk = [0,1,2,3] 
    indl = [0,1,2,3]
    
    nskip = 1 #args.input_length
    Nseq  = (args.train_length-args.input_length - args.npred+1) // nskip
    Ntraj = len(indk) * len(indl)
    Ns = Ntraj * Nseq

    args.iters = Ns // 10
    args.dt = dt[0]
    train_set   = torch.zeros(args.input_length + args.npred,        Ns, 2*args.nred+2*args.nvar, dtype=torch.double)
    samp_set    = torch.zeros(args.input_length,              nsamp, Ns, 2*args.nvar,               dtype=torch.double)
    forcing_set = torch.zeros(args.input_length + args.npred,        Ns, args.nred,               dtype=torch.double)
    dampingm_set = torch.zeros(Ns, args.nmp,args.nred,2, dtype=torch.double)
    dampingv_set = torch.zeros(Ns, args.nvar,2*args.nvar,2, dtype=torch.double)
    for l in range(len(indk)):
        for j in range(len(indl)):
            df = fk[args.indm,indk[l]]
            #print('l = {}, j = {}, df = {}'.format(indk[l], indl[j], df))
            for i in range(Nseq):
                ind = i+Nseq*j+len(indl)*Nseq*l
                samp_set[:,:, ind, :args.nvar]         = torch.from_numpy( np.transpose(zt_r[:,i*nskip:i*nskip + args.input_length,:nsamp, indk[l],indl[j]], (1,2,0)))
                samp_set[:,:, ind, args.nvar:]         = torch.from_numpy( np.transpose(zt_i[:,i*nskip:i*nskip + args.input_length,:nsamp, indk[l],indl[j]], (1,2,0)))
                train_set[:, ind, :args.nmp]          = torch.from_numpy( np.transpose(um_r[:, i*nskip:i*nskip + args.input_length + args.npred,  indk[l],indl[j]], (1,0)))
                train_set[:, ind, args.nmp:args.nred] = torch.from_numpy( np.transpose(um_i[1:,i*nskip:i*nskip + args.input_length + args.npred,  indk[l],indl[j]], (1,0)))
                train_set[:, ind, args.nred:args.nred+args.nvar] = torch.from_numpy( np.transpose(var[:,i*nskip:i*nskip + args.input_length + args.npred,  indk[l],indl[j]], (1,0)))
                train_set[:, ind, args.nred+args.nvar] = torch.from_numpy( theta_m0[i*nskip:i*nskip + args.input_length + args.npred,  indk[l],indl[j]])
                train_set[:, ind, args.nred+args.nvar+1:args.nred+args.nvar+args.nmp] = torch.from_numpy( np.transpose(theta_mr[:,i*nskip:i*nskip + args.input_length + args.npred,  indk[l],indl[j]], (1,0)))
                train_set[:, ind, args.nred+args.nvar+args.nmp:2*args.nred+args.nvar] = torch.from_numpy( np.transpose(theta_mi[:,i*nskip:i*nskip + args.input_length + args.npred,  indk[l],indl[j]], (1,0)))
                train_set[:, ind, 2*args.nred+args.nvar:] = torch.from_numpy( np.transpose(theta_v[:,i*nskip:i*nskip + args.input_length + args.npred,  indk[l],indl[j]], (1,0)))
                
                forcing_set[:,ind, :args.nmp] = torch.from_numpy(df.real)[None,:]
                forcing_set[:,ind, args.nmp:] = torch.from_numpy(df[1:].imag)[None,:]
                dampingm_set[ind,:,:,0]         = torch.from_numpy(Dkm[:,:,indl[j]].real)
                dampingm_set[ind,:,:,1]         = torch.from_numpy(Dkm[:,:,indl[j]].imag)
                dampingv_set[ind,:,:,0]         = torch.from_numpy(Dkv[:,:,indl[j]].real)
                dampingv_set[ind,:,:,1]         = torch.from_numpy(Dkv[:,:,indl[j]].imag)
           
        
    train_loader = (train_set,samp_set, forcing_set,dampingm_set,dampingv_set, vareq)
    del(data_load,um,um_r,um_i,var, zt,zt_r,zt_i, theta_mk,theta_mr,theta_mi, theta_v)
    

    # training performance measure
    epoch_loss = np.zeros((args.epoch, 2))
    epoch_accu   = np.zeros((args.epoch,args.nred, 2))
    epoch_accr   = np.zeros((args.epoch,args.nvar, 2))
    epoch_accthv = np.zeros((args.epoch,args.nvar, 2))
    epoch_accthm = np.zeros((args.epoch,args.nred, 2))
    for epoch in range(ires, args.epoch):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [{} | {}] LR: {:.8f} loss: {}'.format(epoch + 1, cfg['epoch'], cfg['lr'], cfg['loss_type']))
        train_loss,vloss, train_accu,vaccu, train_accr,vaccr, train_accthv,vaccthv, \
        train_accthm,vaccthm,  pred, gt = train(train_loader, model, criterion, optimizer, dev)

        # save accuracy
        epoch_loss[epoch,0]  = train_loss
        epoch_accu[epoch,:,0]  = train_accu
        epoch_accthm[epoch,:,0]  = train_accthm
        epoch_accr[epoch,:,0] = train_accr
        epoch_accthv[epoch,:,0] = train_accthv
        epoch_loss[epoch,1]  = vloss
        epoch_accu[epoch,:,1]  = vaccu
        epoch_accthm[epoch,:,1]  = vaccthm
        epoch_accr[epoch,:,1] = vaccr
        epoch_accthv[epoch,:,1] = vaccthv
        
        # append logger file
        logger.append([epoch, cfg['lr'], train_loss, train_accu[0],train_accu[1:].sum(),train_accr.sum(), 
                       train_accthv.sum(), train_accthm[0],train_accthm[1:].sum()])
        filepath1 = os.path.join(cfg['checkpoint'], 'modelm0_' + fname)
        torch.save({'model_state_dict': model_m0.state_dict(), 
                    'optimizer_state_dict': optim_m0.state_dict(),}, filepath1)
        filepath2 = os.path.join(cfg['checkpoint'], 'modelmk_' + fname)
        torch.save({'model_state_dict': model_mk.state_dict(), 
                    'optimizer_state_dict': optim_mk.state_dict(),}, filepath2)
        filepath3 = os.path.join(cfg['checkpoint'], 'modelrv_' + fname)
        torch.save({'model_state_dict': model_rv.state_dict(), 
                    'optimizer_state_dict': optim_rv.state_dict(),}, filepath3)

        
    datapath = os.path.join(cfg['checkpoint'], 'train_' + fname)
    np.savez(datapath, tt = tt, epoch_loss = epoch_loss, epoch_accu = epoch_accu,epoch_accr = epoch_accr, 
             epoch_accthv = epoch_accthv, epoch_accthm = epoch_accthm, pred = pred, gt = gt) 

    logger.close()
    
    
def train(train_loader, model, criterion, optimizer, device):
    model_m0, model_mk, model_rv = model
    optim_m0, optim_mk, optim_rv = optimizer
    model_m0.train()
    model_mk.train()
    model_rv.train()
    l96_inte = l96_sys.l96_inhomo_split(dt=args.dt, indm=args.indm,indr=args.indr, coeff=args.coeff, coeff_r=args.coeff_cr,coeff_i=args.coeff_ci, device=device)
    
    batch_time = AverageMeter()
    losses     = AverageMeter()
    accsu      = AverageMeter()
    accsr      = AverageMeter()
    accsthv    = AverageMeter()
    accsthm    = AverageMeter()
    end = time.time()
    
    input_full,samp_full, forcing_full,dampingm_full,dampingv_full, vareq = train_loader
    veqm_red = vareq[args.indm]
    veqv_red = vareq[args.indr]
    veq0 = vareq[0]
    veqm_red = veqm_red.to(device)
    veqv_red = veqv_red.to(device)
    veq0 = veq0.to(device)
    dsize = args.iters
    s_idx = random.sample(range(0,input_full.size(1)), dsize)
    stat_iter    = input_full[:, s_idx,:].pin_memory()
    samp_iter    = samp_full[:,:,s_idx,:].pin_memory()
    forcing_iter = forcing_full[:, s_idx,:].pin_memory()
    dampingm_iter = dampingm_full[s_idx,:,:,:].pin_memory()
    dampingv_iter = dampingv_full[s_idx,:,:,:].pin_memory()
    
    noptim = args.nopt
    optim_m0.zero_grad()
    optim_mk.zero_grad()
    optim_rv.zero_grad()  # zero the gradient buffers
    loss = 0.
    for ib in range(0, args.iters):
        inputs_m  =    stat_iter[:-1,ib:ib+1, :].to(device, non_blocking=True)
        inputs_s  =    samp_iter[:,:, ib, :].to(device, non_blocking=True)
        targets_m =    stat_iter[1:,ib:ib+1, :].to(device, non_blocking=True)
        forcing   = forcing_iter[:,ib:ib+1,:].to(device, non_blocking=True)
        damping_m   = dampingm_iter[ib:ib+1,:,:,:].to(device, non_blocking=True)
        damping_v   = dampingv_iter[ib:ib+1,:,:,:].to(device, non_blocking=True)

        # iteration the model in npred steps
        hidden_m0, hidden_mk, hidden_v = (), (), ()
        istate_m0 = torch.empty(args.input_length, 1, 1+args.nred+args.nvar,         dtype=torch.double, device=device)
        istate_mk = torch.empty(args.input_length, 1, args.nmnk+args.nred+args.nvar, dtype=torch.double, device=device)
        istate_v  = torch.empty(args.input_length, 1, args.nred+2*args.nvar,         dtype=torch.double, device=device)
        z_in      = torch.empty(args.input_length, args.train_batch, 2*args.nvar,      dtype=torch.double, device=device)
        r_in      = torch.empty(args.input_length, 1, args.nvar,                     dtype=torch.double, device=device)
        # assign input data
        istate_m0[:,:,:] = inputs_m[:args.input_length,:,:1+args.nred+args.nvar]
        istate_mk[:,:,:args.nred] = inputs_m[:args.input_length,:,:args.nred]
        istate_mk[:,:,args.nred+args.nvar:] = inputs_m[:args.input_length,:,1+args.nred+args.nvar:2*args.nred+args.nvar]
        istate_v[:,:,:args.nred]  = inputs_m[:args.input_length,:,:args.nred]
        istate_v[:,:,args.nred+args.nvar:]  = inputs_m[:args.input_length,:,2*args.nred+args.nvar:]
        z_in[:,:,:]       = inputs_s[:args.input_length,:,:]
        
        r_comp = z_in.var(1)[:,None,:]
        # r_in[:,:,0] = r_comp[:,:,0]
        r_in[:,:,:] = r_comp[:,:,:args.nvar]+r_comp[:,:,args.nvar:]
        # update the variance using sample output
        istate_m0[:,:,args.nred:args.nred+args.nvar] = r_in.clone()
        istate_mk[:,:,args.nred:args.nred+args.nvar] = r_in.clone()
        istate_v[:,:, args.nred:args.nred+args.nvar] = r_in.clone()
        
        pred = torch.empty(args.npred, 2*args.nred+2*args.nvar, dtype=torch.double, device=device)
        for ip in range(args.npred):
            # update the mean model state
            th0_out, hidden_m0 = model_m0(istate_m0, hidden_m0, device=device)
            thk_out, hidden_mk = model_mk(istate_mk, hidden_mk, device=device)
            th0_out = th0_out[:,:,0]
            u0_out, uk_out = l96_inte.dyn_mean_u(istate_m0,r_in, th0_out,thk_out, forcing[ip:ip+args.input_length+1],damping_m)
            
            # run model in one forward iteration
            qm_out,  hidden_v = model_rv(istate_v, hidden_v, device=device)
            dm_out = -torch.min(qm_out, torch.tensor([-1e-3], dtype=torch.double,device=device)) / veqv_red.view(1,1,-1).expand_as(qm_out) #r_in
            sm_out =  torch.max(qm_out, torch.tensor([1e-3],  dtype=torch.double,device=device))
            thv_out = -dm_out*r_in + sm_out
            
            
            # forward euler
            z_out = l96_inte.dyn_coeff_z(istate_m0,z_in, dm_out,sm_out, damping_v)
            r_comp = z_out.var(1)[:,None,:]
            r_out  = torch.empty(args.input_length, 1, args.nvar, dtype=torch.double, device=device)
            # r_out[:,:,0] = r_comp[:,:,0]
            r_out[:,:,:] = r_comp[:,:,:args.nvar]+r_comp[:,:,args.nvar:]
            
            pred[ip, 0]                             = u0_out[-1]
            pred[ip, 1:args.nred]                   = uk_out[-1]
            pred[ip, args.nred:args.nred+args.nvar] = r_out[-1]
            pred[ip, args.nred+args.nvar]           = th0_out[-1]
            pred[ip, args.nred+args.nvar+1:2*args.nred+args.nvar] = thk_out[-1]
            pred[ip, 2*args.nred+args.nvar:]        = thv_out[-1]
            
            if ip < args.npred-1:
                istate_m0 = torch.empty_like(istate_m0)
                istate_mk = torch.empty_like(istate_mk)
                istate_v = torch.empty_like(istate_v)
                z_in = torch.empty_like(z_in)
                r_in = torch.empty_like(r_in)
                # update with full model output
                istate_m0[:,:,0]    = u0_out.clone()
                istate_m0[:,:,1:args.nred]    = uk_out.clone()
                istate_m0[:,:,args.nred:args.nred+args.nvar] = r_out.clone()
                istate_m0[:,:,-1]   = th0_out.clone()
                istate_mk[:,:,0]    = u0_out.clone()
                istate_mk[:,:,1:args.nred]    = uk_out.clone()
                istate_mk[:,:,args.nred:args.nred+args.nvar] = r_out.clone()
                istate_mk[:,:,args.nred+args.nvar:] = thk_out.clone()
                istate_v[:,:,0]    = u0_out.clone()
                istate_v[:,:,1:args.nred]    = uk_out.clone()
                istate_v[:,:,args.nred:args.nred+args.nvar] = r_out.clone()
                istate_v[:,:,args.nred+args.nvar:]  = thv_out.clone()
                z_in[:,:,:]        = z_out.clone()
                r_in[:,:,:]        = r_out.clone()

            output = torch.transpose(torch.cat([u0_out[:,:,None],uk_out[:,:,:],r_out[:,:,:],th0_out[:,:,None],thk_out[:,:,:], thv_out[:,:,:]],2), 0,1)
            target = torch.transpose(targets_m[ip:args.input_length+ip,:,:], 0,1)
            if args.loss_type == 'state':
                crion1, crion2 = criterion
                out_m0 = output[:, -args.nloss:, 0] 
                tag_m0 = target[:, -args.nloss:, 0] 
                out_mk = output[:, -args.nloss:, 1:args.nred] 
                tag_mk = target[:, -args.nloss:, 1:args.nred]
                out_r = output[:, -args.nloss:, args.nred:args.nvar+args.nred] 
                tag_r = target[:, -args.nloss:, args.nred:args.nvar+args.nred] 
                loss += 1.*crion2(out_m0, tag_m0)+1.*crion1(out_r, tag_r) + 1.*crion2(out_mk, tag_mk)
            elif args.loss_type == 'comb':
                crion1, crion2 = criterion
                out_m0 = output[:, -args.nloss:, 0] 
                tag_m0 = target[:, -args.nloss:, 0] 
                out_mk = output[:, -args.nloss:, 1:args.nred] 
                tag_mk = target[:, -args.nloss:, 1:args.nred]
                out_r = output[:, -args.nloss:, args.nred:args.nvar+args.nred] 
                tag_r = target[:, -args.nloss:, args.nred:args.nvar+args.nred] 
                loss += 1.*crion2(out_m0, tag_m0)+1.*crion1(out_r, tag_r) + 1.*crion2(out_mk, tag_mk)
                        
                out_fm0 = output[:, -args.nloss:, args.nvar+args.nred] 
                tag_fm0 = target[:, -args.nloss:, args.nvar+args.nred] 
                out_fmk = output[:, -args.nloss:, 1+args.nvar+args.nred:args.nvar+2*args.nred] 
                tag_fmk = target[:, -args.nloss:, 1+args.nvar+args.nred:args.nvar+2*args.nred]
                out_fr = output[:, -args.nloss:, args.nvar+2*args.nred:] 
                tag_fr = target[:, -args.nloss:, args.nvar+2*args.nred:]
                loss += 1.*crion2(out_fm0, tag_fm0)+1.*crion1(out_fr, tag_fr) + 1.*crion2(out_fmk, tag_fmk)
            elif args.loss_type == 'kld':
                crion1, crion2 = criterion
                out_m0 = output[:, -args.nloss:, 0] / (veq0.sqrt())
                tag_m0 = target[:, -args.nloss:, 0] / (veq0.sqrt())
                out_mk1 = output[:, -args.nloss:, 1:args.nmp] / ((veqm_red[1:]/2).sqrt())
                tag_mk1 = target[:, -args.nloss:, 1:args.nmp] / ((veqm_red[1:]/2).sqrt())
                out_mk2 = output[:, -args.nloss:, args.nmp:args.nred] / ((veqm_red[1:]/2).sqrt())
                tag_mk2 = target[:, -args.nloss:, args.nmp:args.nred] / ((veqm_red[1:]/2).sqrt())
                out_r = output[:, -args.nloss:, args.nred:args.nvar+args.nred] / veqv_red.view(1,1,-1)
                tag_r = target[:, -args.nloss:, args.nred:args.nvar+args.nred] / veqv_red.view(1,1,-1)
                loss += 1.*crion2(out_m0, tag_m0) + 1.*crion2(out_r, tag_r) \
                       +1.*crion2(out_mk1, tag_mk1) + 1.*crion2(out_mk2, tag_mk2)


        losses.update(loss.item() )
        if (ib+1) % noptim == 0 or ib == (args.iters-1):
            loss = loss / noptim
            loss.backward()
            optim_m0.step()
            optim_mk.step()
            optim_rv.step()
            
            optim_m0.zero_grad()
            optim_mk.zero_grad()
            optim_rv.zero_grad()  # zero the gradient buffers
            loss = 0.
        
        # get trained output
        pred_out = pred
        gt_out   = targets_m[args.input_length-1:,0,:]
        accsu.update(   ((pred_out[:,:args.nred]-gt_out[:,:args.nred]).square().mean(0)).data.cpu().numpy() ) 
        accsr.update(   ((pred_out[:,args.nred:args.nvar+args.nred] - \
                            gt_out[:,args.nred:args.nvar+args.nred]).square().mean(0)).data.cpu().numpy() )
        accsthm.update( ((pred_out[:,args.nvar+args.nred:args.nvar+2*args.nred] - \
                            gt_out[:,args.nvar+args.nred:args.nvar+2*args.nred]).square().mean(0)).data.cpu().numpy() )
        accsthv.update( ((pred_out[:,args.nvar+2*args.nred:]-gt_out[:,args.nvar+2*args.nred:]).square().mean(0) ).data.cpu().numpy() )
        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        
        #LR = optimizer.param_groups[0]['lr']
        suffix = 'Iter{iter}: Loss = {loss:.5e} u = {accu:.5e} R = {accr:.5e} thv = {accthv:.5e} thm = {accthm:.5e}  run_time = {bt:.2f}'.format( 
                  iter = ib, loss = losses.val, accu=accsu.val.sum(), accr=accsr.val.sum(), accthv=accsthv.val.sum(), 
                             accthm=accsthm.val.sum(), bt = batch_time.sum)
        if ib % 50 == 0:
            print(suffix)
            
    # get trained output
    pred = pred_out.data.cpu().numpy()
    gt   = gt_out.data.cpu().numpy()

    return losses.avg,losses.var, accsu.avg,accsu.var, accsr.avg,accsr.var, accsthv.avg,accsthv.var, \
           accsthm.avg,accsthm.var, pred, gt
 
def adjust_learning_rate(optimizer, epoch):
    global cfg
    if epoch in cfg['schedule']:
        cfg['lr'] *= args.gamma
        for ioptim in optimizer:
            for param_group in ioptim.param_groups:
                param_group['lr'] = cfg['lr'] 
  

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    gflags.DEFINE_boolean('pretrained', False, 'Use pretrained model')
    gflags.DEFINE_boolean('eval', True, 'Run tests with the network')
    
    main(pretrained = gflags.FLAGS.pretrained, valid = gflags.FLAGS.eval)
