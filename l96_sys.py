"""
Forward Euler integrator for mean L-96 equation
"""
import numpy as np
# import numpy.matlib as matlib
import torch

__all__ = ['l96_mean', 'l96_implicit', 'l96_2mom', 'l96_red', 'l96_coeff', 'l96_inhomo', 'l96_inhomo_split']

class l96_mean(object):
    "compute the one-step forward integration for the mean equation of L-96 system"
    def __init__(self, dt, nm = 40, damp = 1., device='cpu'):
        self.dt = dt
        self.damp = damp
        self.J = nm
    def mean_euler(self, inputs, theta, forcing):
        # nseq  = inputs.shape[0]
        # nsamp = inputs.shape[1]
        u_in = inputs[:,:,0]
        E_in = inputs[:,:,1]
        
        FU = theta - self.damp * u_in + forcing
        u_out = u_in + FU * self.dt 
        FE = -2*self.damp * E_in + forcing * u_in *40/self.J
        E_out = E_in + FE * self.dt
        return u_out, E_out
    
class l96_implicit(object):
    "compute the one-step forward integration for the mean equation of L-96 system"
    def __init__(self, dt, nm = 40., damp = 1., device='cpu'):
        self.dt = dt
        self.damp = damp
        self.J = nm
    def mean_euler(self, inputs, theta, forcing):
        u_in = inputs[:,:,0]
        E_in = inputs[:,:,1]
        
        FU = theta - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *40/self.J
        E_out = inv_E * (E_in + FE * self.dt)
        return u_out, E_out
    
class l96_2mom(object):
    "compute the one-step forward integration for the mean equation of L-96 system"
    def __init__(self, dt, coeff, nm = 40, damp = 1., device='cpu'):
        self.dt = dt
        self.damp = damp
        self.J = nm
        self.coeff = torch.from_numpy(coeff).to(device, non_blocking=True)
        
        dk = np.empty((coeff.shape[0],1))
        dk[[0,-1],:] = coeff[[0,-1],:]
        dk[1:-1,:] = coeff[1:-1,:]/2
        self.dk = torch.from_numpy(dk).to(device, non_blocking=True)
    def mean_euler(self, inputs, forcing):
        u_in = inputs[:,:,0]
        v_in = inputs[:,:,1:22]
        E_in = inputs[:,:,22]
        
        theta = (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2)
        FU = theta - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *40/self.J
        E_out = inv_E * (E_in + FE * self.dt)
        return u_out, E_out
    def mean_midpt(self, inputs,v_out, forcing):
        u_in = inputs[:,:,0]
        v_in = inputs[:,:,1:22]
        
        theta_m  = (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2) / 2
        theta_m += (self.coeff.view(1,1,-1).expand_as(v_out)*v_out).sum(2) / 2
        FU = theta_m - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        return u_out, theta_m
    def mean_ene(self, inputs,v_out, forcing):
        u_in = inputs[:,:,0]
        v_in = inputs[:,:,1:22]
        E_in = inputs[:,:,22]
        
        theta_m = (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2) / 2
        theta_m += (self.coeff.view(1,1,-1).expand_as(v_out)*v_out).sum(2) / 2
        FU = theta_m - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *40/self.J
        E_out = inv_E * (E_in + FE * self.dt)
        return u_out, theta_m, E_out
    def var_mean_ene(self, inputs,v_in, theta_v, forcing):
        u_in = inputs[:,:,0]
        E_in = inputs[:,:,1]

        theta_m = (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2) 
        FU = theta_m - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *40/self.J
        E_out = inv_E * (E_in + FE * self.dt)
        
        FV = theta_v + (-self.damp - u_in[:,:,None]*self.dk.view(1,1,-1)) * v_in
        inv_v = 1/(1+self.dt*(self.damp + u_out[:,:,None]*self.dk.view(1,1,-1)) )
        v_out = inv_v * (v_in + FV * self.dt)
        return u_out, theta_m, E_out, v_out
    def var_mean_clos(self, inputs,v_in, dmk,smk, forcing):
        u_in = inputs[:,:,0]
        E_in = inputs[:,:,1]

        theta_m = (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2) 
        FU = theta_m - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *40/self.J
        E_out = inv_E * (E_in + FE * self.dt)
        
        FV = -dmk*v_in + smk + (-self.damp - u_in[:,:,None]*self.dk.view(1,1,-1)) * v_in
        inv_v = 1/(1+self.dt*(self.damp + u_out[:,:,None]*self.dk.view(1,1,-1)) )
        v_out = inv_v * (v_in + FV * self.dt)
        return u_out, theta_m, E_out, v_out
    
class l96_red(object):
    "compute the one-step forward integration for the mean equation of L-96 system"
    def __init__(self, dt, coeff, k1=0,k2=21, nm = 5, damp = 1., device='cpu'):
        self.dt = dt
        self.damp = damp
        self.J = nm
        self.k1 = k1
        self.k2 = k2
        self.coeff = torch.from_numpy(coeff[k1:k2]).to(device, non_blocking=True)
        
        dk = np.empty((coeff.shape[0],1))
        dk[[0,-1],:] = coeff[[0,-1],:]
        dk[1:-1,:] = coeff[1:-1,:]/2
        self.dk = torch.from_numpy(dk[k1:k2]).to(device, non_blocking=True)
    def mv_ene_red(self, inputs,v_in, th_m,th_v, forcing):
        u_in = inputs[:,:,0]
        E_in = inputs[:,:,1]

        theta_m = th_m + (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2) 
        FU = theta_m - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *40/self.J
        E_out = inv_E * (E_in + FE * self.dt)
        
        FV = th_v + (-self.damp - u_in[:,:,None]*self.dk.view(1,1,-1)) * v_in
        inv_v = 1/(1+self.dt*(self.damp + u_out[:,:,None]*self.dk.view(1,1,-1)) )
        v_out = inv_v * (v_in + FV * self.dt)
        return u_out, E_out, v_out
    def mvar_ene_red(self, inputs,v_in, th_m,th_v, forcing):
        u_in = inputs[:,:,0]
        E_in = inputs[:,:,1]

        theta_m = th_m + (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2) 
        FU = theta_m - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *40/self.J
        E_out = inv_E * (E_in + FE * self.dt)
        
        FV = th_v + (-self.damp - u_in[:,:,None]*self.dk.view(1,1,-1)) * v_in
        inv_v = 1/(1+self.dt*(self.damp + u_out[:,:,None]*self.dk.view(1,1,-1)) )
        v_out = inv_v * (v_in + FV * self.dt)
        return u_out, theta_m, E_out, v_out
    def mean_ene_red(self, inputs,v_in, th_m, forcing):
        u_in = inputs[:,:,0]
        E_in = inputs[:,:,1]

        theta_m = th_m + (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2) 
        FU = theta_m - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *1/self.J
        E_out = inv_E * (E_in + FE * self.dt)

        return u_out, E_out, theta_m
    
class l96_coeff(object):
    "compute the one-step forward integration for the stochastic coefficient equation of L-96 system"
    def __init__(self, dt, coeff_r,coeff_i, k1=0,k2=21, nm = 5, damp = 1., device='cpu'):
        self.dt = dt
        self.damp = damp
        self.J = nm
        self.k1 = k1
        self.k2 = k2
        self.kred = k2-k1
        self.coeff_r = torch.from_numpy(coeff_r[k1:k2]).to(device, non_blocking=True)
        self.coeff_i = torch.from_numpy(coeff_i[k1:k2]).to(device, non_blocking=True)
        
        self.device = device
    def coeff_dyn(self, inputs,z_in, th_z):
        u_in  = inputs[:,:,0]
        #u_out = inputs[1:,:,0]
        # E_in = inputs[:,:,1]
        
        nseq  = z_in.shape[0]
        nsamp = z_in.shape[1]
        z_fl = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        z_fl[:,:,:self.kred] = -z_in[:,:,self.kred:]
        z_fl[:,:,self.kred:] = z_in[:,:,:self.kred]

        coeff1 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff2 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff1[:self.kred] = self.coeff_r
        coeff1[self.kred:] = self.coeff_r
        coeff2[:self.kred] = self.coeff_i
        coeff2[self.kred:] = self.coeff_i
        
        FZ = th_z - self.damp*z_in - (u_in[:,:,None]*coeff1.view(1,1,-1)) * z_in \
                                     - (u_in[:,:,None]*coeff2.view(1,1,-1)) * z_fl 
        z_out = z_in + FZ * self.dt
        return z_out
    def coeff_dyn_impl(self, inputs,z_in, th_z):
        u_in  = inputs[:-1,:,0]
        u_out = inputs[1:,:,0]
        # E_in = inputs[:,:,1]
        
        nseq  = z_in.shape[0]
        nsamp = z_in.shape[1]
        z_fl = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        z_fl[:,:,:self.kred] = -z_in[:,:,self.kred:]
        z_fl[:,:,self.kred:] = z_in[:,:,:self.kred]

        coeff1 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff2 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff1[:self.kred] = self.coeff_r
        coeff1[self.kred:] = self.coeff_r
        coeff2[:self.kred] = self.coeff_i
        coeff2[self.kred:] = self.coeff_i
        
        FZ = th_z - self.damp*z_in/2 - (u_in[:,:,None]*coeff1.view(1,1,-1)) * z_in/2 \
                                     - (u_in[:,:,None]*coeff2.view(1,1,-1)) * z_fl/2 
        z_incre = z_in + FZ * self.dt

        in_fl = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        in_fl[:,:,:self.kred] =  z_incre[:,:,self.kred:]
        in_fl[:,:,self.kred:] = -z_incre[:,:,:self.kred]

        a = 1 + self.dt*(self.damp/2 + u_out[:,:,None]*self.coeff_r.view(1,1,-1)/2)
        b =     self.dt*(              u_out[:,:,None]*self.coeff_i.view(1,1,-1)/2)
        abn = a**2+b**2
        coeff3 = torch.empty(nseq,nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        coeff4 = torch.empty(nseq,nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        coeff3[:,:,:self.kred] = a/abn
        coeff3[:,:,self.kred:] = a/abn
        coeff4[:,:,:self.kred] = b/abn
        coeff4[:,:,self.kred:] = b/abn

        z_out = coeff3*z_incre + coeff4*in_fl
        return z_out
    def coeff_dyn_stat(self, inputs,z_in, th_z):
        u_in  = inputs[:-1,:,0]
        u_out = inputs[1:,:,0]
        # E_in = inputs[:,:,1]
        
        nseq  = z_in.shape[0]
        nsamp = z_in.shape[1]
        z_fl = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        z_fl[:,:,:self.kred] = -z_in[:,:,self.kred:]
        z_fl[:,:,self.kred:] = z_in[:,:,:self.kred]

        coeff1 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff2 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff1[:self.kred] = self.coeff_r
        coeff1[self.kred:] = self.coeff_r
        coeff2[:self.kred] = self.coeff_i
        coeff2[self.kred:] = self.coeff_i
        
        FZ = th_z - self.damp*z_in/2 - (u_in[:,:,None]*coeff1.view(1,1,-1)) * z_in/2 \
                                     - (u_in[:,:,None]*coeff2.view(1,1,-1)) * z_fl/2 
        z_incre = z_in + FZ * self.dt

        in_fl = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        in_fl[:,:,:self.kred] =  z_incre[:,:,self.kred:]
        in_fl[:,:,self.kred:] = -z_incre[:,:,:self.kred]

        a = 1 + self.dt*(self.damp/2 + u_out[:,:,None]*self.coeff_r.view(1,1,-1)/2)
        b =     self.dt*(              u_out[:,:,None]*self.coeff_i.view(1,1,-1)/2)
        abn = a**2+b**2
        coeff3 = torch.empty(nseq,nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        coeff4 = torch.empty(nseq,nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        coeff3[:,:,:self.kred] = a/abn
        coeff3[:,:,self.kred:] = a/abn
        coeff4[:,:,:self.kred] = b/abn
        coeff4[:,:,self.kred:] = b/abn

        z_out = coeff3*z_incre + coeff4*in_fl
        return z_out
    def coeff_dyn_closure(self, inputs,z_in, d_z,s_z):
        u_in  = inputs[:-1,:,0]
        u_out = inputs[1:,:,0]
        
        nseq  = z_in.shape[0]
        nsamp = z_in.shape[1]
        z_fl = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        z_fl[:,:,:self.kred] = -z_in[:,:,self.kred:]
        z_fl[:,:,self.kred:] = z_in[:,:,:self.kred]

        coeff1 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff2 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff1[:self.kred] = self.coeff_r
        coeff1[self.kred:] = self.coeff_r
        coeff2[:self.kred] = self.coeff_i
        coeff2[self.kred:] = self.coeff_i
        
        damp_z = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        sigm_z = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        damp_z[:,:,:self.kred] = d_z/2
        damp_z[:,:,self.kred:] = d_z/2
        sigm_z[:,:,:self.kred] = (s_z/2).sqrt()
        sigm_z[:,:,self.kred:] = (s_z/2).sqrt()
        Wz = torch.randn(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        
        th_z = -damp_z/2*z_in
        FZ = th_z - self.damp*z_in/2 - (u_in[:,:,None]*coeff1.view(1,1,-1)) * z_in/2 \
                                     - (u_in[:,:,None]*coeff2.view(1,1,-1)) * z_fl/2 
        z_incre = z_in + FZ * self.dt + np.sqrt(self.dt) * sigm_z * Wz

        in_fl = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        in_fl[:,:,:self.kred] =  z_incre[:,:,self.kred:]
        in_fl[:,:,self.kred:] = -z_incre[:,:,:self.kred]

        a = 1 + self.dt*(self.damp/2 + d_z/4 + u_out[:,:,None]*self.coeff_r.view(1,1,-1)/2)
        b =     self.dt*(              u_out[:,:,None]*self.coeff_i.view(1,1,-1)/2)
        abn = a**2+b**2
        coeff3 = torch.empty(nseq,nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        coeff4 = torch.empty(nseq,nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        coeff3[:,:,:self.kred] = a/abn
        coeff3[:,:,self.kred:] = a/abn
        coeff4[:,:,:self.kred] = b/abn
        coeff4[:,:,self.kred:] = b/abn

        z_out = coeff3*z_incre + coeff4*in_fl
        return z_out
    def coeff_dyn_euler(self, inputs,z_in, d_z,s_z):
        u_in  = inputs[:-1,:,0]
        #u_out = inputs[1:,:,0]
        # E_in = inputs[:,:,1]
        
        nseq  = z_in.shape[0]
        nsamp = z_in.shape[1]
        z_fl = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        z_fl[:,:,:self.kred] = -z_in[:,:,self.kred:]
        z_fl[:,:,self.kred:] = z_in[:,:,:self.kred]

        coeff1 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff2 = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        coeff1[:self.kred] = self.coeff_r
        coeff1[self.kred:] = self.coeff_r
        coeff2[:self.kred] = self.coeff_i
        coeff2[self.kred:] = self.coeff_i
        
        damp_z = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        sigm_z = torch.empty(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        damp_z[:,:,:self.kred] = d_z/2
        damp_z[:,:,self.kred:] = d_z/2
        sigm_z[:,:,:self.kred] = (s_z/2).sqrt()
        sigm_z[:,:,self.kred:] = (s_z/2).sqrt()
        Wz = torch.randn(nseq, nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        
        th_z = -damp_z*z_in
        FZ = th_z - self.damp*z_in - (u_in[:,:,None]*coeff1.view(1,1,-1)) * z_in \
                                   - (u_in[:,:,None]*coeff2.view(1,1,-1)) * z_fl 
        z_out = z_in + FZ * self.dt + np.sqrt(self.dt) * sigm_z * Wz
        return z_out
    
    
class l96_inhomo(object):
    "compute the one-step forward integration for the inhomogeneous equation of L-96 system"
    def __init__(self, dt, indr, coeff,coeff_r,coeff_i, device='cpu'):
        self.dt = dt
        self.indr = indr
        self.nvar = len(indr)
        self.nred = 2*len(indr)-1
        
        self.coeff = torch.from_numpy(coeff[indr]).to(device, non_blocking=True)
        self.coeff_r = torch.from_numpy(coeff_r[indr]).to(device, non_blocking=True)
        self.coeff_i = torch.from_numpy(coeff_i[indr]).to(device, non_blocking=True)
        
        self.device = device
    def dyn_mean_u(self, inputs,v_in, th0_m,thk_m, forcing,damping):
        u0_in = inputs[:,:,0]
        uk_in = inputs[:,:,1:self.nred]
        
        nseq  = u0_in.shape[0]
        nsamp = u0_in.shape[1]
        dk1 = torch.cat(( damping[0,:,:, 0],damping[0,1:,:, 1]), 0).to(self.device, non_blocking=True)
        dk2 = torch.cat((-damping[0,:,1:,1],damping[0,1:,1:,0]), 0).to(self.device, non_blocking=True)
        u1 = torch.zeros(nseq, nsamp, self.nred,   dtype=torch.double, device=self.device)
        u2 = torch.zeros(nseq, nsamp, self.nred-1, dtype=torch.double, device=self.device)
        u1[:,:,:self.nvar] = inputs[:,:,:self.nvar]
        u1[:,:,self.nvar:] = inputs[:,:,1:self.nvar]
        u2[:,:,:self.nvar-1] = inputs[:,:,self.nvar:self.nred]
        u2[:,:,self.nvar-1:] = -inputs[:,:,self.nvar:self.nred]
        damp_u = torch.matmul(dk1, u1[:,:,:,None]) + torch.matmul(dk2, u2[:,:,:,None])
        damp_u = damp_u[:,:,:,0]
        
        # damp_u = torch.empty(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        # u1 = torch.zeros(self.nred,   dtype=torch.double, device=self.device)
        # u2 = torch.zeros(self.nred-1, dtype=torch.double, device=self.device)
        # for i in range(nseq):
        #     for j in range(nsamp):
        #         dk_r = damping[j,:,:,0].to(self.device, non_blocking=True)
        #         dk_i = damping[j,:,:,1].to(self.device, non_blocking=True)
        #         u1[:self.nvar] = inputs[i,j,:self.nvar]
        #         u1[self.nvar:] = inputs[i,j,1:self.nvar]
        #         u2[:self.nvar-1] = inputs[i,j,self.nvar:self.nred]
        #         u2[self.nvar-1:] = -inputs[i,j,self.nvar:self.nred]
        #         damp_u[i,j,:self.nvar] = torch.matmul(dk_r,u1)     - torch.matmul(dk_i[:,1:],u2)
        #         damp_u[i,j,self.nvar:] = torch.matmul(dk_i[1:,:],u1) + torch.matmul(dk_r[1:,1:],u2)
        
        um2 = inputs[:,:,:self.nvar].square()
        um2[:,:,1:] = um2[:,:,1:] + inputs[:,:,self.nvar:self.nred].square()
        ene1 = (self.coeff.view(1,1,-1).expand_as(um2)*um2).sum(2) 
        ene2 = (self.coeff.view(1,1,-1).expand_as(v_in)*v_in).sum(2)

        theta_m0 = th0_m + ene1 + ene2
        FU0 = theta_m0 - damp_u[:,:,0] + (forcing[:-1,:,0]+forcing[1:,:,0])/2
        u0_out = u0_in + FU0 * self.dt
        
        uk1 = inputs[:,:,1:self.nvar]
        uk2 = inputs[:,:,self.nvar:self.nred]
        mcrx = torch.empty(nseq, nsamp, self.nred-1,   dtype=torch.double, device=self.device)
        mcrx[:,:,:self.nvar-1]  = self.coeff_r[1:].view(1,1,-1).expand_as(uk1)*uk1*u0_in[:,:,None]
        mcrx[:,:,:self.nvar-1] -= self.coeff_i[1:].view(1,1,-1).expand_as(uk2)*uk2*u0_in[:,:,None]
        mcrx[:,:,self.nvar-1:]  = self.coeff_i[1:].view(1,1,-1).expand_as(uk1)*uk1*u0_in[:,:,None]
        mcrx[:,:,self.nvar-1:] += self.coeff_r[1:].view(1,1,-1).expand_as(uk2)*uk2*u0_in[:,:,None]
        theta_mk = thk_m - mcrx
        FUk = theta_mk - damp_u[:,:,1:] + (forcing[:-1,:,1:]+forcing[1:,:,1:])/2
        uk_out = uk_in + FUk * self.dt

        return u0_out, uk_out
    def dyn_coeff_z(self, inputs,z_in, d_z,s_z, damping):
        u0_in = inputs[:,:,0]
        uk_in = inputs[:,:,1:self.nred]
        
        nseq  = z_in.shape[0]
        nsamp = z_in.shape[1]
        z_fl = torch.zeros(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        z_fl[:,:,1:self.nvar] = -z_in[:,:,self.nvar:]
        z_fl[:,:,self.nvar:] = z_in[:,:,1:self.nvar]

        coeff1 = torch.empty(self.nred, dtype=torch.double, device=self.device)
        coeff2 = torch.empty(self.nred, dtype=torch.double, device=self.device)
        coeff1[:self.nvar] = self.coeff_r
        coeff1[self.nvar:] = self.coeff_r[1:]
        coeff2[:self.nvar] = self.coeff_i
        coeff2[self.nvar:] = self.coeff_i[1:]
        
        dk1 = torch.cat(( damping[0,:,:, 0],damping[0,1:,:, 1]), 0).to(self.device, non_blocking=True)
        dk2 = torch.cat((-damping[0,:,1:,1],damping[0,1:,1:,0]), 0).to(self.device, non_blocking=True)
        z1 = torch.zeros(nseq, nsamp, self.nred,   dtype=torch.double, device=self.device)
        z2 = torch.zeros(nseq, nsamp, self.nred-1, dtype=torch.double, device=self.device)
        z1[:,:,:self.nvar] = z_in[:,:,:self.nvar]
        z1[:,:,self.nvar:] = z_in[:,:,1:self.nvar]
        z2[:,:,:self.nvar-1] = z_in[:,:,self.nvar:self.nred]
        z2[:,:,self.nvar-1:] = -z_in[:,:,self.nvar:self.nred]
        dZ = torch.matmul(dk1, z1[:,:,:,None]) + torch.matmul(dk2, z2[:,:,:,None])
        dZ = dZ[:,:,:,0]
        
        # dZ = torch.empty(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        # for i in range(nseq):
        #     for j in range(nsamp):
        #         dk_r = damping[0,:,:,0].to(self.device, non_blocking=True)
        #         dk_i = damping[0,:,:,1].to(self.device, non_blocking=True)
        #         z1[:self.nvar]   =  z_in[i,j,:self.nvar]
        #         z1[self.nvar:]   =  z_in[i,j,1:self.nvar]
        #         z2[:self.nvar-1] =  z_in[i,j,self.nvar:self.nred]
        #         z2[self.nvar-1:] = -z_in[i,j,self.nvar:self.nred]
        #         dZ[i,j,:self.nvar] = torch.matmul(dk_r,z1)     - torch.matmul(dk_i[:,1:],z2)
        #         dZ[i,j,self.nvar:] = torch.matmul(dk_i[1:,:],z1) + torch.matmul(dk_r[1:,1:],z2)
                
        
        damp_z = torch.empty(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        sigm_z = torch.empty(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        damp_z[:,:,0] = d_z[:,:,0]/2
        damp_z[:,:,1:self.nvar] = d_z[:,:,1:]/2
        damp_z[:,:,self.nvar:]  = d_z[:,:,1:]/2
        sigm_z[:,:,0] = (s_z[:,:,0]).sqrt()
        sigm_z[:,:,1:self.nvar] = (s_z[:,:,1:]/2).sqrt()
        sigm_z[:,:,self.nvar:]  = (s_z[:,:,1:]/2).sqrt()
        Wz = torch.randn(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        
        th_z = -damp_z*z_in
        FZ = th_z - dZ - (u0_in[:,:,None]*coeff1.view(1,1,-1)) * z_in \
                       - (u0_in[:,:,None]*coeff2.view(1,1,-1)) * z_fl 
        z_out = z_in + FZ * self.dt + np.sqrt(self.dt) * sigm_z * Wz
        return z_out
    def dyn_coeff_cov_z(self, inputs,z_in, d_z,s_z, damping):
        u0_in = inputs[:,:,0]
        uk_in = inputs[:,:,1:self.nred]
        
        nseq  = z_in.shape[0]
        nsamp = z_in.shape[1]
        z_fl = torch.zeros(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        z_fl[:,:,1:self.nvar] = -z_in[:,:,self.nvar:]
        z_fl[:,:,self.nvar:] = z_in[:,:,1:self.nvar]

        coeff1 = torch.empty(self.nred, dtype=torch.double, device=self.device)
        coeff2 = torch.empty(self.nred, dtype=torch.double, device=self.device)
        coeff1[:self.nvar] = self.coeff_r
        coeff1[self.nvar:] = self.coeff_r[1:]
        coeff2[:self.nvar] = self.coeff_i
        coeff2[self.nvar:] = self.coeff_i[1:]
        
        dk1 = torch.cat(( damping[0,:,:, 0],damping[0,1:,:, 1]), 0).to(self.device, non_blocking=True)
        dk2 = torch.cat((-damping[0,:,1:,1],damping[0,1:,1:,0]), 0).to(self.device, non_blocking=True)
        z1 = torch.zeros(nseq, nsamp, self.nred,   dtype=torch.double, device=self.device)
        z2 = torch.zeros(nseq, nsamp, self.nred-1, dtype=torch.double, device=self.device)
        z1[:,:,:self.nvar] = z_in[:,:,:self.nvar]
        z1[:,:,self.nvar:] = z_in[:,:,1:self.nvar]
        z2[:,:,:self.nvar-1] = z_in[:,:,self.nvar:self.nred]
        z2[:,:,self.nvar-1:] = -z_in[:,:,self.nvar:self.nred]
        dZ = torch.matmul(dk1, z1[:,:,:,None]) + torch.matmul(dk2, z2[:,:,:,None])
        dZ = dZ[:,:,:,0]
                
        
        damp_z = torch.empty(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        damp_z[:,:,:self.nvar] = torch.matmul(d_z[:,:,:,:].real, z_in[:,:,:self.nvar,None])[:,:,:,0] - \
                                 torch.matmul(d_z[:,:,:,1:].imag, z_in[:,:,self.nvar:,None])[:,:,:,0]
        damp_z[:,:,self.nvar:] = torch.matmul(d_z[:,:,1:,1:].real, z_in[:,:,self.nvar:,None])[:,:,:,0] + \
                                 torch.matmul(d_z[:,:,1:,:].imag, z_in[:,:,:self.nvar,None])[:,:,:,0]
        
        sigm_z = torch.empty(nseq, nsamp, self.nred, dtype=torch.double, device=self.device)
        Wz1 = torch.randn(nseq, nsamp, self.nvar, dtype=torch.double, device=self.device) / np.sqrt(2)
        Wz2 = torch.randn(nseq, nsamp, self.nvar-1, dtype=torch.double, device=self.device) / np.sqrt(2)
        sigm_z[:,:,:self.nvar] = torch.matmul(s_z.real, Wz1[:,:,:,None])[:,:,:,0] - \
                                 torch.matmul(s_z[:,:,:,1:].imag, Wz2[:,:,:,None])[:,:,:,0]
        sigm_z[:,:,self.nvar:] = torch.matmul(s_z[:,:,1:,1:].real, Wz2[:,:,:,None])[:,:,:,0] + \
                                 torch.matmul(s_z[:,:,1:,:].imag, Wz1[:,:,:,None])[:,:,:,0]
        
        FZ = -damp_z - dZ - (u0_in[:,:,None]*coeff1.view(1,1,-1)) * z_in \
                       - (u0_in[:,:,None]*coeff2.view(1,1,-1)) * z_fl 
        z_out = z_in + FZ * self.dt + np.sqrt(self.dt) * sigm_z
        return z_out
    

class l96_inhomo_split(object):
        "compute the one-step forward integration for the inhomogeneous equation of L-96 system"
        def __init__(self, dt, indm,indr, coeff,coeff_r,coeff_i, device='cpu'):
            self.dt = dt
            self.indm = indm
            self.indr = indr
            self.nmp  = len(indm)
            self.nvar = len(indr)
            self.nred = 2*len(indm)-1
            
            self.coeff = torch.from_numpy(coeff[:]).to(device, non_blocking=True)
            self.coeff_r = torch.from_numpy(coeff_r[:]).to(device, non_blocking=True)
            self.coeff_i = torch.from_numpy(coeff_i[:]).to(device, non_blocking=True)
            
            self.device = device
        def dyn_mean_u(self, inputs,v_in, th0_m,thk_m, forcing,damping):
            u0_in = inputs[:,:,0]
            uk_in = inputs[:,:,1:self.nred]
            
            nseq  = u0_in.shape[0]
            nsamp = u0_in.shape[1]
            dk1 = torch.cat(( damping[0,:,:, 0],damping[0,1:,:, 1]), 0).to(self.device, non_blocking=True)
            dk2 = torch.cat((-damping[0,:,1:,1],damping[0,1:,1:,0]), 0).to(self.device, non_blocking=True)
            u1 = torch.zeros(nseq, nsamp, self.nred,   dtype=torch.double, device=self.device)
            u2 = torch.zeros(nseq, nsamp, self.nred-1, dtype=torch.double, device=self.device)
            u1[:,:,:self.nmp] = inputs[:,:,:self.nmp]
            u1[:,:,self.nmp:] = inputs[:,:,1:self.nmp]
            u2[:,:,:self.nmp-1] = inputs[:,:,self.nmp:self.nred]
            u2[:,:,self.nmp-1:] = -inputs[:,:,self.nmp:self.nred]
            damp_u = torch.matmul(dk1, u1[:,:,:,None]) + torch.matmul(dk2, u2[:,:,:,None])
            damp_u = damp_u[:,:,:,0]
            
            um2 = inputs[:,:,:self.nmp].square()
            um2[:,:,1:] = um2[:,:,1:] + inputs[:,:,self.nmp:self.nred].square()
            ene1 = (self.coeff[self.indm].view(1,1,-1).expand_as(um2)*um2).sum(2) 
            ene2 = (self.coeff[self.indr].view(1,1,-1).expand_as(v_in)*v_in).sum(2)

            theta_m0 = th0_m + ene1 + ene2
            FU0 = theta_m0 - damp_u[:,:,0] + (forcing[:-1,:,0]+forcing[1:,:,0])/2
            u0_out = u0_in + FU0 * self.dt
            
            uk1 = inputs[:,:,1:self.nmp]
            uk2 = inputs[:,:,self.nmp:self.nred]
            mcrx = torch.empty(nseq, nsamp, self.nred-1,   dtype=torch.double, device=self.device)
            mcrx[:,:,:self.nmp-1]  = self.coeff_r[self.indm[1:]].view(1,1,-1).expand_as(uk1)*uk1*u0_in[:,:,None]
            mcrx[:,:,:self.nmp-1] -= self.coeff_i[self.indm[1:]].view(1,1,-1).expand_as(uk2)*uk2*u0_in[:,:,None]
            mcrx[:,:,self.nmp-1:]  = self.coeff_i[self.indm[1:]].view(1,1,-1).expand_as(uk1)*uk1*u0_in[:,:,None]
            mcrx[:,:,self.nmp-1:] += self.coeff_r[self.indm[1:]].view(1,1,-1).expand_as(uk2)*uk2*u0_in[:,:,None]
            theta_mk = thk_m - mcrx
            FUk = theta_mk - damp_u[:,:,1:] + (forcing[:-1,:,1:]+forcing[1:,:,1:])/2
            uk_out = uk_in + FUk * self.dt

            return u0_out, uk_out
        def dyn_coeff_z(self, inputs,z_in, d_z,s_z, damping):
            u0_in = inputs[:,:,0]
            uk_in = inputs[:,:,1:self.nred]
            
            nseq  = z_in.shape[0]
            nsamp = z_in.shape[1]
            z_fl = torch.zeros(nseq, nsamp, 2*self.nvar, dtype=torch.double, device=self.device)
            z_fl[:,:,:self.nvar] = -z_in[:,:,self.nvar:]
            z_fl[:,:,self.nvar:] = z_in[:,:,:self.nvar]

            coeff1 = torch.empty(2*self.nvar, dtype=torch.double, device=self.device)
            coeff2 = torch.empty(2*self.nvar, dtype=torch.double, device=self.device)
            coeff1[:self.nvar] = self.coeff_r[self.indr]
            coeff1[self.nvar:] = self.coeff_r[self.indr]
            coeff2[:self.nvar] = self.coeff_i[self.indr]
            coeff2[self.nvar:] = self.coeff_i[self.indr]
            
            dk1 = torch.cat(( damping[0,:,:, 0],damping[0,:,:, 1]), 0).to(self.device, non_blocking=True)
            dk2 = torch.cat((-damping[0,:,:,1], damping[0,:,:,0]), 0).to(self.device, non_blocking=True)
            z1 = torch.zeros(nseq, nsamp, 2*self.nvar, dtype=torch.double, device=self.device)
            z2 = torch.zeros(nseq, nsamp, 2*self.nvar, dtype=torch.double, device=self.device)
            z1[:,:,:self.nvar] = z_in[:,:,:self.nvar]
            z1[:,:,self.nvar:] = z_in[:,:,:self.nvar]
            z2[:,:,:self.nvar] = z_in[:,:, self.nvar:]
            z2[:,:,self.nvar:] = -z_in[:,:,self.nvar:]
            dZ = torch.matmul(dk1, z1[:,:,:,None]) + torch.matmul(dk2, z2[:,:,:,None])
            dZ = dZ[:,:,:,0]
                    
            
            damp_z = torch.empty(nseq, nsamp, 2*self.nvar, dtype=torch.double, device=self.device)
            sigm_z = torch.empty(nseq, nsamp, 2*self.nvar, dtype=torch.double, device=self.device)
            # damp_z[:,:,0] = d_z[:,:,0]/2
            damp_z[:,:,:self.nvar] = d_z[:,:,:]/2
            damp_z[:,:,self.nvar:] = d_z[:,:,:]/2
            # sigm_z[:,:,0] = (s_z[:,:,0]).sqrt()
            sigm_z[:,:,:self.nvar] = (s_z[:,:,:]/2).sqrt()
            sigm_z[:,:,self.nvar:] = (s_z[:,:,:]/2).sqrt()
            Wz = torch.randn(nseq, nsamp, 2*self.nvar, dtype=torch.double, device=self.device)
            
            th_z = -damp_z*z_in
            FZ = th_z - dZ - (u0_in[:,:,None]*coeff1.view(1,1,-1)) * z_in \
                           - (u0_in[:,:,None]*coeff2.view(1,1,-1)) * z_fl 
            z_out = z_in + FZ * self.dt + np.sqrt(self.dt) * sigm_z * Wz
            return z_out

