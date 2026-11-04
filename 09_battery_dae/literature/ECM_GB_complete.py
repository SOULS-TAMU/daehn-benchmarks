# -*- coding: utf-8 -*-
"""
Grey-box model of a lithium-ion battery on the basis of an 
equivalent circuit model using neural ordinary differential equations
part 2: the complete model

measurement data: 
CALB home storage cell    

author: Jennifer Brucker
date: December 14, 2021


MIT License
Copyright (c) 2021 Jennifer Brucker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
"""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))

import pandas as pd
from scipy import interpolate
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint


# Loading the csv files with OCV-SOC data
df = pd.read_csv('OCV_SOC_data.csv', sep=";")
SOC = torch.DoubleTensor(df.iloc[:,0])
OCV = torch.DoubleTensor(df.iloc[:,1])

OCV_SOC = interpolate.interp1d(SOC.squeeze(), OCV.squeeze(), kind='linear', axis=0, bounds_error=False, fill_value=(OCV[0],OCV[-1]))
SOC_OCV = interpolate.interp1d(OCV.squeeze(), SOC.squeeze(), kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')


# data preparation
def data_prep(index):
    df = pd.read_csv(str(index), sep=";")    
    tsim = torch.DoubleTensor(df.iloc[:,0])
    tsim = tsim - tsim[0]
    i = torch.DoubleTensor(df.iloc[:,1])
    I= interpolate.interp1d(tsim.squeeze(), i.squeeze(), kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
    u = torch.DoubleTensor(df.iloc[:,2])
    soc0 = SOC_OCV(u[0])

    return tsim, i, I, u, soc0


# loading the csv files with the measurement data
# CCCV measurement data
tsim1, i1, I1, u1, soc1 = data_prep('CCCV_18A_dis.csv')
tsim2, i2, I2, u2, soc2 = data_prep('CCCV_18A_chg.csv')
tsim3, i3, I3, u3, soc3 = data_prep('CCCV_50A_dis.csv')
tsim4, i4, I4, u4, soc4 = data_prep('CCCV_50A_chg.csv')
tsim5, i5, I5, u5, soc5 = data_prep('CCCV_180A_dis.csv')
tsim6, i6, I6, u6, soc6 = data_prep('CCCV_180A_chg.csv')


# pulsed current - measurement data
tsim7, i7, I7, u7, soc7 = data_prep('pulsed_dis.csv')
tsim8, i8, I8, u8, soc8 = data_prep('pulsed_chg.csv')

# half cycles - measurement data
tsim9, i9, I9, u9, soc9 = data_prep('halfcycles.csv')

# loadprofile - measurement data
tsim10, i10, I10, u10, soc10 = data_prep('loadprofile.csv')


# Visualization of the results
def visualize(pred_u, true_u, t, pred_SOC, batch_i):
       
    fig, ax = plt.subplots(1,3)
    
    ax[0].plot(t, batch_i.numpy())
    ax[0].set_xlabel('$t/\mathrm{s}$')
    ax[0].set_ylabel('$i_\mathrm{bat}/\mathrm{A}$')
    
    ax[1].set_xlabel('$t/\mathrm{s}$')
    ax[1].set_ylabel('SOC')
    ax[1].plot(t, pred_SOC.numpy())
    
    ax[2].plot(t, true_u.numpy(), label='true')
    ax[2].plot(t, pred_u.numpy(), '--', label='learned')
    ax[2].set_xlabel('$t/\mathrm{s}$')
    ax[2].set_ylabel('$u_\mathrm{bat}/\mathrm{V}$')


    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    fig.align_ylabels()
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
    fig.subplots_adjust(bottom=0.1)
    fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)




# selection of training data
def get_batch(s):
    
    # training data
    if s == 0:
        batch_I = torch.DoubleTensor(i7).squeeze()
        batch_u = torch.DoubleTensor(u7).squeeze()
        batch_t = torch.DoubleTensor(tsim7).squeeze()
        batch_soc0 = torch.DoubleTensor(soc7).squeeze()
        batch_ifunc = I7  

    elif s == 1:
        batch_I = torch.DoubleTensor(i8).squeeze()
        batch_u = torch.DoubleTensor(u8).squeeze()
        batch_t = torch.DoubleTensor(tsim8).squeeze()
        batch_soc0 = torch.DoubleTensor(soc8).squeeze()
        batch_ifunc = I8
        
    if s == 2:
        batch_I = torch.DoubleTensor(i1).squeeze()
        batch_u = torch.DoubleTensor(u1).squeeze()
        batch_t = torch.DoubleTensor(tsim1).squeeze()
        batch_soc0 = torch.DoubleTensor(soc1).squeeze()
        batch_ifunc = I1
        
    elif s == 3:
        batch_I = torch.DoubleTensor(i2).squeeze()
        batch_u = torch.DoubleTensor(u2).squeeze()
        batch_t = torch.DoubleTensor(tsim2).squeeze()
        batch_soc0 = torch.DoubleTensor(soc2).squeeze()
        batch_ifunc = I2
        
    elif s == 4:
        batch_I = torch.DoubleTensor(i3).squeeze()
        batch_u = torch.DoubleTensor(u3).squeeze()
        batch_t = torch.DoubleTensor(tsim3).squeeze()
        batch_soc0 = torch.DoubleTensor(soc3).squeeze()
        batch_ifunc = I3
       
    elif s == 5:
        batch_I = torch.DoubleTensor(i4).squeeze()
        batch_u = torch.DoubleTensor(u4).squeeze()
        batch_t = torch.DoubleTensor(tsim4).squeeze()
        batch_soc0 = torch.DoubleTensor(soc4).squeeze()
        batch_ifunc = I4

    elif s == 6:
        batch_I = torch.DoubleTensor(i5).squeeze()
        batch_u = torch.DoubleTensor(u5).squeeze()
        batch_t = torch.DoubleTensor(tsim5).squeeze()
        batch_soc0 = torch.DoubleTensor(soc5).squeeze()
        batch_ifunc = I5
        
    elif s == 7:
        batch_I = torch.DoubleTensor(i6).squeeze()
        batch_u = torch.DoubleTensor(u6).squeeze()
        batch_t = torch.DoubleTensor(tsim6).squeeze()
        batch_soc0 = torch.DoubleTensor(soc6).squeeze()
        batch_ifunc = I6
        
    # test data 
    elif s == 8:
        batch_I = torch.DoubleTensor(i9).squeeze()
        batch_u = torch.DoubleTensor(u9).squeeze()
        batch_t = torch.DoubleTensor(tsim9).squeeze()
        batch_soc0 = torch.DoubleTensor(soc9).squeeze()
        batch_ifunc = I9  

    elif s == 9:
        batch_I = torch.DoubleTensor(i10).squeeze()
        batch_u = torch.DoubleTensor(u10).squeeze()
        batch_t = torch.DoubleTensor(tsim10).squeeze()
        batch_soc0 = torch.DoubleTensor(soc10).squeeze()
        batch_ifunc = I10
        
        
    # provide the current functions to the module
    func3.i = batch_ifunc

    return batch_I, batch_u, batch_t, batch_soc0.reshape(-1,1)



# dSOC/dt and dU_RC1/dt
class ODEFunc(nn.Module):
    def __init__(self, i):
        super(ODEFunc, self).__init__()
        
        self.i = i

        self.nl = nn.ReLU()
  
        l = 100 # number of hidden neurons  
        
 
        self.F = nn.Sequential(
            nn.Linear(2,l).double(),
            self.nl,
            nn.Linear(l,1).double()
            )

        self.G = nn.Sequential(
            nn.Linear(2,l).double(),
            self.nl,
            nn.Linear(l,1).double()
            )

               
        self.C1 = nn.Linear(1,1, bias = False).double()

        # initialization of C
        K = torch.DoubleTensor([191.5])
        self.CN = torch.nn.Parameter(K)
        
    def forward(self, t, x):

        # i(t)
        ifunc = self.i
        i = torch.DoubleTensor(ifunc(t.detach().numpy())).reshape(-1,1)

        if torch.abs(i)<0.25: 
            i = torch.DoubleTensor([0]).reshape(-1,1)

        # dSOC/dt   
        SOC = -1/(self.CN*3600) * i
        
        # SOC(t)
        soc = x[1,:].reshape(-1,1)
        
        # C_1
        C = torch.abs(self.C1(torch.DoubleTensor([1]))) 

        i_soc = torch.cat((i/180, soc), dim = 1)

        if i > 0:
            R1 = torch.abs(self.F(i_soc))    
        elif i < 0: 
            R1 = torch.abs(self.G(i_soc)) 
        else: 
            R1 = 1/2 * (torch.abs(self.F(i_soc)) + torch.abs(self.G(i_soc)))
            

        # dU_RC1/dt
        URC1 = 1 / ( R1 * 1000 * C ) * x[0,:]   # scaling of R_1 and C_1
        URC2 = 1 / (100000*C) * i # scaling of C_1

        URC =  (URC2 - URC1) 

        return torch.cat((URC, SOC), dim = 0)




# R_S and v_hys 
class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()

        self.faktor = nn.Linear(2, 1, bias = False).double()
        
        # initialization of R_S and v_hys
        K = torch.DoubleTensor([[0.28,0.15]]) 
        self.faktor.weight = torch.nn.Parameter(K)
                    
        
    def forward(self, x):  
        
        out = self.faktor(x)
                               
        return out       




def calculations(s):
    
   
    # data set selection    
    batch_I, batch_u, batch_t, soc0 = get_batch(s)

    # pre-processing of the battery current 
    for index in range(len(batch_I)):
        if torch.abs(batch_I[index]) < 0.25:
            batch_I[index] = 0
            
    # calculate the SOC and U_RC1   
    urc10 = torch.zeros(1).reshape(-1,1) # intial U_RC1 value
    batch_0 = torch.cat((urc10, soc0), dim = 0) # initial SOC and U_RC1 values
    
    # solving the differential equation system 
    # higher absolute tolerance for the half cycles 
    if s == 8:
        pred_SOC_URC1 = odeint(func3, batch_0.reshape(-1,1), batch_t.squeeze(), method = 'dopri8', rtol=1e-3, atol = 1e-3)
    else: 
        pred_SOC_URC1 = odeint(func3, batch_0.reshape(-1,1), batch_t.squeeze(), method = 'dopri8', rtol=1e-3, atol = 1e-5)

        
    pred_URC1 = pred_SOC_URC1[:,0].squeeze()
    pred_SOC = pred_SOC_URC1[:,1].squeeze()

    # calculate OCV
    pred_OCV = torch.zeros(len(batch_t))
    
    for i in range(len(batch_t)):
        pred_OCV[i] = torch.DoubleTensor(OCV_SOC(pred_SOC[i].detach().numpy()))
    
    # calculate the voltage drop across R_S and the hysteresis voltage drop   
    signum_i = torch.zeros(batch_I.size())  
    for ii in range(len(batch_I)):
        if batch_I[ii] !=0:
            signum_i[ii] = batch_I[ii]/abs(batch_I[ii])

    U_R_in = torch.stack((batch_I/1000, signum_i/10), dim=1)  # scaling of R_S and v_hys
    pred_U = func4(U_R_in).squeeze()
    
    
    
    # calculate the battery voltage
    pred_U_cell =  pred_OCV - pred_URC1.squeeze() - pred_U
    
    
    # definition of the loss function
    loss = lossfun(pred_U_cell, batch_u)**0.5
    
    # the SOC should stay in the range of 0 to 1 
    for i in range(len(batch_t)):
        if pred_SOC[i] > 1: 
            loss = loss + 100* pred_SOC[i]
        elif pred_SOC[i] < 0:
            loss = loss - 100* (pred_SOC[i] - 1)
    
    return loss, pred_U_cell, batch_u, batch_t, batch_I, pred_SOC
    





if __name__ == '__main__':

    # Initialization
    func3 = ODEFunc(I1)
    func4 = FFNet()
    
    lossfun = nn.MSELoss()
    
    
    # loading pretrained parameters
    # former func0 and func1 are now combined in func3 
    ECM_func1_2_R = torch.load(('func1_params.pt'))
    ECM_func1_2_R.update(torch.load(('func0_params.pt')))
    torch.save(ECM_func1_2_R, 'zw.pt')          
    
    func3.load_state_dict(torch.load('zw.pt'), strict = False)
    
    # func4 is identical to func2 of the static network
    func4.load_state_dict(torch.load('func2_params.pt'))
    

    # initialization of C_1
    K = torch.DoubleTensor([[0.5]])
    func3.C1.weight = torch.nn.Parameter(K)
    

    params = (list(func3.parameters()) + list(func4.parameters()))
    
    optimizer = optim.Adam(list(func3.C1.parameters()), lr=1e-3)

    loss0 = np.inf


    # Training
    for itr in range(1, 30 + 1):    # 30 training epochs

        # adjustment of the learnable parameters
        if itr == 20:
            optimizer = optim.Adam(params, lr=1e-3)

            
        optimizer.zero_grad()
    
    
        # random order of training data
        # in the first ten epochs only the data from measurements with pulsed currents is taken into account
        if itr < 10:
            s = np.random.choice((2), 2, replace=False)
            
        else: 
            s = np.random.choice((8), 8, replace=False)
         
        if itr == 10: 
            loss0 = np.inf
            
        loss_gesamt = 0 


        for i in range(len(s)):
            
            loss, pred_U, batch_u, batch_t, batch_I, pred_SOC = calculations(s[i])
            
            
            # visualization
            if itr % 5 == 0 or itr == 1 or itr >20:
                visualize(pred_U.detach().squeeze(), batch_u.detach().squeeze(), batch_t.detach(), pred_SOC.detach(), batch_I.detach())     
                plt.show() 
                print('loss: ',loss.item())
                print('C: ', func3.CN)
                print('C_1: ', func3.C1.weight)
                print('R_S, v_hys: ', func4.faktor.weight)


            
            loss_gesamt = loss_gesamt + loss
            
        # save the model parameters            
        if loss_gesamt<loss0:
            loss0 = loss_gesamt
            torch.save(func3.state_dict(), 'func3_params.pt')
            torch.save(func4.state_dict(), 'func4_params.pt')
            print('saving model')
            
        # backpropagation and optimization of learnable parameters # batch gradient descent 
        loss_gesamt.backward()
        optimizer.step()
        optimizer.zero_grad()   
            
        
        print('Epoch' , itr, ' finished')
        
    print('\n training complete')    
    ############### training complete ################################################### 
        
    # load the final model parameters    
    func3.load_state_dict(torch.load('func3_params.pt'))
    func4.load_state_dict(torch.load('func4_params.pt'))
    
    # print the final learned parameters
    print('C: ', func3.CN)   
    print('C_1: ', func3.C1.weight)
    print('R_S, v_hys: ', func4.faktor.weight)

    # test
    with torch.no_grad():
        # half cycles
        loss, pred_U, batch_u, batch_t, batch_I, pred_SOC = calculations(8)
        visualize(pred_U.detach().squeeze(), batch_u.detach().squeeze(), batch_t.detach(), pred_SOC.detach(), batch_I.detach())     
        plt.show()
        print('loss: ',loss.item())
        
        # loadprofile
        loss, pred_U, batch_u, batch_t, batch_I, pred_SOC = calculations(9)
        visualize(pred_U.detach().squeeze(), batch_u.detach().squeeze(), batch_t.detach(), pred_SOC.detach(), batch_I.detach())     
        plt.show()
        print('loss: ',loss.item())

        
