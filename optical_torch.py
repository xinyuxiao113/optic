import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
import numpy as np

def complex_leakyRelu(input):
    return F.leaky_relu(input.real,negative_slope=0.01) + F.leaky_relu(input.imag,negative_slope=0.01)*(1j)

def apply_complex(fr, fi, input, dtype = torch.complex64):
    '''
    operation in complex form
    (fr + i fr) (x + iy) = fr(x) - fi(y) + [fr(x) + fi(y)]i
    '''
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class complex_linear(nn.Module):
    '''
    Complex linear mapping:
    y = W x + b
    W \in C^{n x m}
    b \in C^{m}
    near zero initilization
    '''
    def __init__(self,input_dim,output_dim):
        super(complex_linear,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        w = torch.randn(self.output_dim, self.input_dim)/100
        v = torch.randn(self.output_dim, self.input_dim)/100
        b0 = torch.randn(self.output_dim)/100
        b1 = torch.randn(self.output_dim)/100
        self.weight = nn.Parameter(torch.complex(w,v))
        self.bias = nn.Parameter(torch.complex(b0,b1))
    
    def forward(self,x):
        return F.linear(x,self.weight,self.bias)


class complex_conv1d(nn.Module):
    '''
    complex con1d mapping
    y = conv1d(W, x) + b
    W \in C^{}
    Input: [B, L, C]
    '''
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding = 0, padding_mode='circular',
                 dilation=1, groups=1, bias=True):
        super(complex_conv1d, self).__init__()
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        
    def forward(self,input):    
        return apply_complex(self.conv_r, self.conv_i, input.transpose(1,2)).transpose(1,2)


class FNN(nn.Module):
    '''
    Fully connected complex network:
    width: a positive integer
    '''
    def __init__(self, input_features,out_features, width=60, depth=2, init_value=torch.zeros(1), to_real=False):
        super(FNN,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.depth= depth
        self.init_value = torch.tensor(init_value)
        self.to_real = to_real
        self.fc0 = complex_linear(input_features, self.width)
        self.net = nn.ModuleList([complex_linear(self.width,self.width) for i in range(self.depth)])
        self.fc1 = complex_linear(self.width,out_features)
        self.activation = complex_leakyRelu

    def forward(self,u):
        u = self.activation(self.fc0(u))
        for net in self.net:
            u = self.activation(net(u))
        u = self.fc1(u)
        if self.to_real:
            u = torch.abs(u)**2
        return u + self.init_value
        #return u



class MetaFiber(nn.Module):
    '''
    Fiber model. SSFM Algorithm.
    '''
    def __init__(self, steps, Nfft, H, phi, meta_width=60, meta_depth=2,sps=8):
        super(MetaFiber,self).__init__()

        # transform to the right device: cpu or gpu
        self.Nfft = Nfft
        self.H = H
        self.phi = phi
        self.steps = steps
        self.H_trained = nn.ModuleList([FNN(self.Nfft, self.Nfft, meta_width, meta_depth,init_value=self.H) for i in range(self.steps)])
        self.scales = nn.ModuleList([FNN(self.Nfft*2,1,meta_width, meta_depth,to_real=True,init_value=torch.ones(1)) for i in range(self.steps)])
        self.conv = complex_conv1d(2,2,sps,sps)

    def lin_step(self,u,step):
        '''
        Linear step
        u: [batch, Nfft, 2]
        '''
        u = torch.fft.ifft(torch.fft.fft(u,dim=1) * self.H_trained[step](torch.mean(u, axis=-1))[...,None], dim=1)

        return u
    
    def nl_step(self,u,step=0):
        '''
        Nonlinear step
        u: [batch, Nfft, 2]
        '''
        power = torch.abs(u)**2
        power = torch.sum(power,dim=-1)
        u = u * torch.exp(-(1j) * self.scales[step](u.reshape(u.shape[0],-1)) * self.phi * power)[...,None]
        return u
    

    def forward(self,u,step=0):
        '''
        SSFM algorithm
        '''
        for step in range(self.steps):
            u = self.lin_step(u,step=step)
            u = self.nl_step(u,step=step)
            
        return self.conv(u)


class NNFiber(nn.Module):
    '''
    Fiber model. SSFM Algorithm.
    '''
    def __init__(self, steps, Nfft, H, phi, meta_width=60, meta_depth=2,sps=8):
        super(NNFiber,self).__init__()

        # transform to the right device: cpu or gpu
        self.Nfft = Nfft
        self.H = H
        self.phi = phi
        self.steps = steps
        self.H_trained = nn.ParameterList([nn.Parameter(self.H) for i in range(self.steps)]) 
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(self.steps)])
        self.conv = complex_conv1d(2,2,sps,sps)

    def lin_step(self,u,step):
        '''
        Linear step
        u: [batch, Nfft, 2]
        '''
        u = torch.fft.ifft(torch.fft.fft(u,dim=1) * self.H_trained[step][...,None], dim=1)

        return u
    
    def nl_step(self,u,step=0):
        '''
        Nonlinear step
        u: [batch, Nfft, 2]
        '''
        power = torch.abs(u)**2
        power = torch.sum(power,dim=-1)
        u = u * torch.exp(-(1j) * self.scales[step] * self.phi * power)[...,None]
        return u
    

    def forward(self,u,step=0):
        '''
        SSFM algorithm
        '''
        for step in range(self.steps):
            u = self.lin_step(u,step=step)
            u = self.nl_step(u,step=step)
            
        return self.conv(u)