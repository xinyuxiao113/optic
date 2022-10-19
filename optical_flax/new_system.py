import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.40'

from jax.config import config
config.update("jax_enable_x64", True)

import pickle, re, os, time
import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt
from functools import partial
from DataLoader import DataLoader,take_data,Generation_DataSet
from optical_flax.fiber_system import get_data, Tx_data
from optical_flax.fiber_tx import  wdm_base
from optical_flax.fiber_channel import edfa, ssfm, manakov_ssf
from optical_flax.layers import MetaSSFM, GRU_DBP, fdbp, NNSSFM,Dense_net
from optical_flax.models import Transformer, CNN, xi_config,H_config
from optical_flax.initializers import fdbp_init, near_zeros,ones
from optical_flax.utils import realize,show_tree
from commplax.module.core import Signal
from commplax import xop
import argparse
import scipy.constants as const
from optical_flax.operator import fft,ifft,fftfreq,fftshift
import jaxlib
DeviceArray = jaxlib.xla_extension.DeviceArray



from flax import struct
from typing import Any, NamedTuple, Iterable, Callable, Optional
from typing import Callable
from commplax.module.core import Signal
Array = Any

from optical_flax.fiber_tx import QAM
from commplax import comm

###### Global parameters 

C = const.c                             # speed of light [m/s]
D = 1.65e-5                             # [s/m^2]
fc = 193414489032258                    # [Hz]
alphaB = 0.0002                         # [dB/m]
gamma = 0.0016567                       # [/W/m]
alpha = alphaB / 10 * np.log(10.)       # [/m]
lamb = C/fc                             # [m]
beta2 = -(D*lamb**2)/(2*np.pi*C)        # [s^2/m]

@struct.dataclass
class MySignal:
    val: Array    # [Nfft,Nmodes]
    Fs: float=struct.field(pytree_node=False)
    sps: int=struct.field(pytree_node=False)
    Nch: int=struct.field(pytree_node=False)
    freqspace: float=struct.field(pytree_node=False)

    def __add__(self,other):
        return  MySignal(val=self.val + other.val, sps=self.sps, Fs=self.Fs, Nch=self.Nch, freqspace=self.freqspace)
    
    def __sub__(self,other):
        return  MySignal(val=self.val - other.val, sps=self.sps, Fs=self.Fs, Nch=self.Nch, freqspace=self.freqspace)


    def __mul__(self,other):
        return  MySignal(val=self.val * other.val, sps=self.sps, Fs=self.Fs, Nch=self.Nch, freqspace=self.freqspace)

    def __truediv__(self,other):
        return  MySignal(val=self.val / other.val, sps=self.sps, Fs=self.Fs, Nch=self.Nch, freqspace=self.freqspace)
    


setting = 'Huawei'

if setting == 'Huawei':
    ## Hua Wei setting
    
    span_length = 80e3                      # [m]
    span_number = 25                        # [1]
    Nch = 19                                # [1]
    power = 3                               # [dBm]
    Rs = 190e9                              # [Hz]
    freqspace = 220e9                       # [Hz]
    sps = 32                                # tx sps
    sps_single = 2                          # cssfm single channel sps
    pulse_taps = 4096
    Nmodes = 2
    Fs = Rs * sps
    Nsymb = 10000
    Nfft = Nsymb*sps


elif setting == 'Toy':
    ## Toy setting
    
    span_length = 100e3                     # [m]
    span_number = 1                         # [1]
    Nch = 3                                 # [1]
    power = 10                              # [dBm]
    Rs = 10e9                               # [Hz]
    freqspace = 50e9                        # [Hz]
    sps = 16                                # tx sps
    sps_single = 2                          # cssfm single channel sps
    Fs = Rs * sps
    Nsymb = 256
    Nmodes=1
    pulse_taps = 4096
    Nfft = Nsymb*sps



def Tx_signal(*args,**kwargs):
    sigWDM, symbWDM, param = Tx_data(*args,**kwargs)
    wdm_signal = MySignal(val=sigWDM, sps=param.SpS, Fs=param.Rs*param.SpS, Nch=param.Nch, freqspace=param.freqSpac)
    return wdm_signal,symbWDM,param

def show_power(E):
    print('mean signal power: %g [W]' % jnp.mean(jnp.abs(E.val)**2))

from functools import wraps
def make_init(f):
    @wraps(f)
    def _f(key, *args, **kwargs):
        return f(*args, **kwargs)
    
    return _f


def wdm_merge(E):
    # E.val.shape  [Nfft,Nch,Nmodes]
    Nfft = E.val.shape[0]
    freqGrid = jnp.arange(-int(E.Nch/2), int(E.Nch/2)+1,1) * E.freqspace
    wdm_wave = wdm_base(Nfft, E.Fs, freqGrid) # [Nfft, Nch]
    x = jnp.sum(E.val * wdm_wave[...,None], axis=-2)
    return MySignal(val=x,sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)



def exp_integral(z: float, alpha:float = alpha, span_length:float=span_length)->DeviceArray:
    '''
       P(z) = exp( -alpha *(z % Lspan))
       exp_integral(z) = int_{0}^{z} P(z) dz
    '''
    k = z // span_length
    z0 = z % span_length

    return k * (1 - jnp.exp(-alpha * span_length)) / alpha + (1- jnp.exp(-alpha * z0)) / alpha


def Leff1(z, dz, alpha=alpha, span_length=span_length):
    ''' 
    split form 1: no normalization
    '''
    return dz

def Leff2(z, dz, alpha=alpha, span_length=span_length):
    '''
    split form 1: normalization
    '''
    return exp_integral(z + dz) - exp_integral(z)

def get_omega(Fs:int, Nfft:int)->DeviceArray:
    ''' 
    get signal fft angular frequency.
    Input:
        Fs: sampling frequency. [Hz]
        Nfft: number of sampling points. 
    Output:
        omega:DeviceArray [Nfft,]
    '''
    return 2*np.pi*Fs*fftfreq(Nfft)


def H1(dz, E):
    ''' 
    split form 1: no normalization

    Input:
        dz: float
        E: MySignal. [Nfft, Nmodes] or [batch, Nfft, Nmodes]
    Output:
        [Nfft]
    '''
    Nfft = E.val.shape[-2]
    omega = get_omega(E.Fs, Nfft)
    return jnp.exp(-alpha/2*dz - 1j * (beta2/2) * (omega**2) * dz)

def H2(dz, E):
    ''' 
    split form 2: normalization

    Input:
        dz: float
        E: MySignal. [Nfft, Nmodes] or [batch, Nfft, Nmodes]
    Output:
        [Nfft]
    '''
    Nfft = E.val.shape[-2]
    omega = get_omega(E.Fs, Nfft)
    return jnp.exp(-1j * (beta2/2) * (omega**2) * dz)


def h1(dz, E, dtaps):
    ''' 
    split form 1: no normalization

    Input:
        dz: float
        E: MySignal. [Nfft, Nmodes]
    Output:
        [dtaps]
    '''
    omega = get_omega(E.Fs, dtaps)
    kernel = jnp.exp(-alpha/2*dz - 1j * (beta2/2) * (omega**2) * dz)
    return fftshift(ifft(kernel))

def h2(dz, E, dtaps):
    ''' 
    split form 2: normalization

    Input:
        dz: float
        E: MySignal. [Nfft, Nmodes]
    Output:
        [Nfft]
    '''
    omega = get_omega(E.Fs, dtaps)
    kernel = jnp.exp(-1j * (beta2/2) * (omega**2) * dz)
    return fftshift(ifft(kernel))



def L2(x):
    return jnp.sqrt(jnp.mean(jnp.abs(x)**2))

def relative_L2(x,y):
    return L2(x-y) / L2(x)

def MSE(x,y):
    return jnp.mean(jnp.abs(x-y)**2)

def test_convergence(E, method, dz,length=80e3):
    Eo = []
    loss = []
    for h in dz:
        Eo.append(method(E, length,h))
    
    for i in range(len(dz)-1):
        loss.append(relative_L2(Eo[i].val, Eo[i+1].val).item())

    return Eo,loss


def rx(E: MySignal, chid:int, new_sps:int) -> MySignal:
    ''' 
    Get single channel information from WDM signal.
    Input:
        E: 1D array. WDM signal. (Nfft,Nmodes)  or  (Nfft, Nch, Nmodes)
        k: channel id.  [0,1,2,...,Nch-1]
        new_sps
    Output:
        E0: single channel signal. (Nfft,Nmodes)
    '''
    assert E.sps % new_sps == 0
    k = chid - E.Nch // 2
    Nfft = E.val.shape[0]
    Fs = E.Fs
    freqspace = E.freqspace
    t = jnp.linspace(0,1/Fs*Nfft, Nfft)
    omega = get_omega(E.Fs, Nfft)
    f = omega/(2*np.pi)
    x0 = ifft(jnp.roll(fft(E.val, axis=0) * (jnp.abs(f - k*freqspace)<freqspace/2)[:,None], -k*int(freqspace/Fs*Nfft), axis=0), axis=0)
    rate = E.sps // new_sps
    return MySignal(val=x0[::rate,:], sps=new_sps, Fs=E.Fs/rate, Nch=E.Nch, freqspace=E.freqspace)

def wdm_split(E:MySignal,  new_sps:int) -> MySignal:
    '''' 
    Get every single channel information from WDM signal.
    Input:
        E: 1D array. WDM signal. (Nfft,Nmodes)
        new_sps: Output sps.
    Output:
        E: single channel signal. (Nfft, Nch, Nmodes)
    '''

    signal = []
    E0 = jax.vmap(rx, in_axes=(None, 0, None), out_axes=1)(E, jnp.arange(E.Nch), new_sps)
    return E0

batch_wdm_split = jax.vmap(wdm_split, in_axes=(0,None), out_axes=0)

def freq(x):
    return jnp.abs(fftshift(fft(x)))





from optical_flax.operator import circFilter

@partial(jax.jit, static_argnums=(3))
def L(E:MySignal, z:float, dz:float, H:Callable=H1) -> MySignal: 
    ''' 
    Linear operator with full FFT convolution.
    Input:
        E: E.val  [Nfft,Nmodes]
        z: operator start position.
        dz: operator distance.
        H: kernel function. [Nfft,]
    Output:
        E: E.val [Nfft, Nmodes]
    '''
    kernel = H(dz, E)   # [Nfft]
    x = ifft(fft(E.val, axis=0) * kernel[:,None], axis=0)
    return MySignal(val=x, sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)


@partial(jax.jit, static_argnums=(3,4))
def Lh(E:MySignal, z:float, dz:float, dtaps:int, H:Callable=h1) -> MySignal: 
    ''' 
    Linear operator with time domain convolution.
    Input:
        E: E.val  [Nfft,Nmodes]
        z: operator start position.
        dz: operator distance.
        dtaps: kernel shape.
        H: kernel function. [dtaps,]
    Output:
        E: E.val [Nfft, Nmodes]
    '''
    kernel = H(dz, E, dtaps)   # [dtaps]
    x = jax.vmap(circFilter, in_axes=(None, -1), out_axes=-1)(kernel, E.val)
    return MySignal(val=x, sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)


@partial(jax.jit, static_argnums=(3))
def NL(E:MySignal, z:float, dz:float, Leff:Callable=Leff1, gamma=gamma) -> MySignal:
    ''' 
    NonLinear operator.
    Input:
        E: E.val  [Nfft,Nmodes]
        z: operator start position.
        dz: operator distance.
        H: kernel function. [Nfft,]
    Output:
        E: E.val [Nfft, Nmodes]
    '''
    phi = gamma * Leff(z, dz) * jnp.sum(jnp.abs(E.val)**2, axis=1)[:,None]
    x = jnp.exp(-(1j)*phi)*E.val
    return MySignal(val=x, sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)



def ssfm1(E: MySignal, length: float, dz: float, H=H1, Leff=Leff1) -> MySignal:
    if type(dz) == int or  type(dz) == float:
        K = int(length / dz)
        dz = np.ones(K) * dz
    else:
        K = len(dz)

    z = 0

    @jax.jit
    def one_step(Ez,h):
        E ,z = Ez
        E = L(E, z, h, H)
        E = NL(E, z, h, Leff)
        return (E, z+h), None
    
    E,z = jax.lax.scan(one_step, (E,z), dz, length=K)[0]

    return E

def ssfm2(Ech: MySignal, length: float, dz:float, H=H1, Leff=Leff1) ->MySignal:
    if type(dz) == int or  type(dz) == float:
        K = int(length / dz)
        dz = np.ones(K) * dz
    else:
        K = len(dz)
    z = 0


    @jax.jit
    def one_step(Ez, h):
        E ,z = Ez
        E = E * H(h/2, Ech)[:,None]
        E = ifft(E, axis=0)
        phi = gamma * Leff(z, h) * jnp.sum(jnp.abs(E)**2, axis=1)[:,None]
        E = jnp.exp(-(1j)*phi)*E
        E = fft(E, axis=0)
        E = E * H(h/2, Ech)[:,None] 
        return (E, z+h), None
    
    E = fft(Ech.val, axis=0)
    E,z = jax.lax.scan(one_step, (E,z), dz, length=K)[0]

    return MySignal(val=E, sps=Ech.sps, Fs=Ech.Fs, Nch=Ech.Nch, freqspace=Ech.freqspace)

def amp1(key, E, span_length):
    '''
    Ideal EDFA
    '''
    x = jnp.exp(1/2*alpha*span_length)*E.val
    return MySignal(val=x, sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)

def amp2(key, E, span_length):
    '''
    identity
    '''
    return E

def amp3(key, E, span_length):
    ''' 
    noisy edfa
    E.val  [sqrt(W)]
    '''
    x = edfa(key, E.val, Fs=E.Fs, G=alphaB*span_length)
    return MySignal(val=x, sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)


def amp3(key, E, span_length):
    ''' 
    noisy edfa. 
    attenuation form.
    E.val  [sqrt(W)]
    '''
    x = edfa(key, E.val, Fs=E.Fs, G=alphaB*span_length)
    return MySignal(val=x, sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)

def amp4(key, E, span_length,NF=4.5):
    ''' 
    noisy edfa.
    Normalization form.
    E.val  [sqrt(W)]
    NF: [dB]
    '''
    NF_lin = 10**(NF/10)
    G = 10**(alphaB * span_length / 10)
    nsp= G * NF_lin/(2*(G-1))
    N_ase = (G-1)*nsp*const.h*fc
    p_noise = N_ase*E.Fs
    noise  = jax.random.normal(key, E.val.shape, dtype=jnp.complex64) * np.sqrt(p_noise)
    x = E.val + noise
    return MySignal(val=x, sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)



def fiber(key, E, dz, spans=20, span_length = 80e3, module=ssfm1, amp=amp2):
    key_full = rd.split(key, spans)
    for i in range(spans):
        E = module(E, span_length, dz)
        E = amp(key_full[i], E, span_length)
        
    return E
