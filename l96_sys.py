"""
Forward Euler integrator for mean L-96 equation
"""
import numpy as np
import torch

__all__ = ['l96_mean',  'l96_2mom', 'l96_red']

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
    def mean_impl(self, inputs, theta_m, forcing):
        u_in = inputs[:,:,0]
        E_in = inputs[:,:,1]

        FU = theta_m - self.damp * u_in/2 + (forcing[:-1]+forcing[1:])/2
        inv_u = 1/(1+self.damp*self.dt/2)
        u_out = inv_u * (u_in + FU * self.dt)
        inv_E = 1/(1+self.damp*self.dt)
        FE = -2*self.damp*E_in/2 + (forcing[:-1]*u_in+forcing[1:]*u_out)/2 *40/self.J
        E_out = inv_E * (E_in + FE * self.dt)
        return u_out, E_out
    
class l96_2mom(object):
    "compute the one-step forward integration for the full 2-mom equation of L-96 system"
    def __init__(self, dt, coeff, nm = 40, damp = 1., device='cpu'):
        self.dt = dt
        self.damp = damp
        self.J = nm
        self.coeff = torch.from_numpy(coeff).to(device, non_blocking=True)
        
        dk = np.empty((coeff.shape[0],1))
        dk[[0,-1],:] = coeff[[0,-1],:]
        dk[1:-1,:] = coeff[1:-1,:]/2
        self.dk = torch.from_numpy(dk).to(device, non_blocking=True)
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

    
class l96_red(object):
    "compute the one-step forward integration for the reduced-order equation of L-96 system"
    def __init__(self, dt, coeff, k1=1,k2=20, nm = 40, damp = 1., device='cpu'):
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
