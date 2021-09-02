import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

__all__ = ['LSTMupdate','LSTMresi']

def add_cell_block(input_size, hidden_size, expan = 1):
    lin = nn.Linear(input_size, expan * hidden_size)
    relu = nn.ReLU(inplace=True)
    
    return [lin, relu]

class inner_cell(nn.Module):
    def __init__(self, input_size, hidden_size, nstages = 0, out_ch = 4):
        """"Inner structure in the LSTM Net"""
        super(inner_cell, self).__init__()
        seq_list = []
        for i in range(nstages):
            if i==0:
                seq_list = add_cell_block(input_size, hidden_size, 1)
            else:
                seq_list += add_cell_block(1 * hidden_size, hidden_size, 1)
           
        if nstages > 0:
            seq_list.append(nn.Linear(1 * hidden_size, out_ch * hidden_size))
        else:
            seq_list.append(nn.Linear(input_size, out_ch * hidden_size))
        self.seq = nn.Sequential(*seq_list)
        
    def forward(self,x):
        return self.seq(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nlayers, nstages = 0):
        """"LSTM Net cell with multiple layers"""
        super(LSTMCell, self).__init__()

        self.hsize = hidden_size
        self.nlayers = nlayers
        
        ih, hh, ch = [], [], []
        hlink = []
        for i in range(nlayers):
            ih.append(inner_cell(input_size,          hidden_size, nstages, 4))
            hh.append(inner_cell((i+1) * hidden_size, hidden_size, nstages, 4))
            ch.append(inner_cell((i+1) * hidden_size, hidden_size, nstages, 3))
            
            hlink.append(nn.Linear(hidden_size, hidden_size))

        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)
        self.w_ch = nn.ModuleList(ch)
        self.hlw = nn.ModuleList(hlink)

    def forward(self, inputs, hidden):
        """"Defines the forward computation of the LSTMCell"""     
        nhy, ncy = hidden[0], hidden[1]
        for i in range(self.nlayers):
            hx, cx = nhy, ncy
            cxi = cx[:, -self.hsize:]
            gates = self.w_ih[i](inputs) + self.w_hh[i](hx)
            peep = self.w_ch[i](cx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            ip_gate, fp_gate, op_gate = peep.chunk(3, 1)
            i_gate = torch.sigmoid(i_gate + ip_gate)
            f_gate = torch.sigmoid(f_gate + fp_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate + op_gate)
            ncx = (f_gate * cxi) + (i_gate * c_gate)
            nhx = o_gate * torch.tanh(ncx)
            
            nhy = torch.cat([nhy, nhx], 1)
            ncy = torch.cat([ncy, ncx], 1)
            
            if i == 0:                
                hout = self.hlw[i](nhx)
            else:
                hout += self.hlw[i](nhx)

        return hout, nhx, ncx

    
class LSTMupdate(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nlayers=4, nstages=0, expan=10):
        super(LSTMupdate, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.nlayers = nlayers
        self.lstmcell = LSTMCell(input_size, hidden_size, output_size, nlayers, nstages)
        self.lin_out = nn.Sequential(nn.Linear(hidden_size, expan * hidden_size),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(expan * hidden_size, output_size))
        
    def forward(self, inputs, hidden = (), npred = 0, device = "cpu"):
        output = []
        for i in range(inputs.size(0)):            
            if i == 0:
                if hidden == ():
                    ht = torch.zeros(inputs.size(1), self.hidden_size, dtype=torch.double, device=device)
                    ct = torch.zeros(inputs.size(1), self.hidden_size, dtype=torch.double, device=device)
                else:
                    ht, ct = hidden
                hout, ht,ct = self.lstmcell(inputs[i], (ht, ct))
            else:
                hout, ht,ct = self.lstmcell(inputs[i], (ht, ct))

            incre = self.lin_out(hout)
            output.append(incre)
                
            if i == npred:
                hidden = (ht, ct)
                
        return torch.stack(output, 0), hidden

class LSTMresi(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nlayers=4, nstages=0, expan=10):
        super(LSTMresi, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.nlayers = nlayers
        self.lstmcell = LSTMCell(input_size, hidden_size, output_size, nlayers, nstages)
        self.lin_out = nn.Sequential(nn.Linear(hidden_size, expan * hidden_size),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(expan * hidden_size, output_size))
        
    def forward(self, inputs, hidden = (), npred = 0, device = "cpu"):
        output = []
        for i in range(inputs.size(0)):            
            if i == 0:
                if hidden == ():
                    ht = torch.zeros(inputs.size(1), self.hidden_size, dtype=torch.double, device=device)
                    ct = torch.zeros(inputs.size(1), self.hidden_size, dtype=torch.double, device=device)
                else:
                    ht, ct = hidden
                hout, ht,ct = self.lstmcell(inputs[i], (ht, ct))
            else:
                hout, ht,ct = self.lstmcell(inputs[i], (ht, ct))

            incre = self.lin_out(hout)
            output.append(inputs[i,:,-self.output_size:] + incre)
                
            if i == npred:
                hidden = (ht, ct)
                
        return torch.stack(output, 0), hidden

    

if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        seq = torch.randn(10, 100, 6, dtype=torch.double)
        lstm = LSTMupdate(seq.size(2), 5, 4, 2).double()
        output_seq, hidden = lstm(seq)
        loss = output_seq.sum()
        loss.backward()
    print(output_seq.size())
