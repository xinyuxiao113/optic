from flax import struct
import scipy.constants as const
import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt
from typing import Any, NamedTuple, Iterable, Callable, Optional
from functools import partial

from optical_flax.operator import fft,ifft,fftfreq,fftshift
Array = Any
DeviceArray = Any

from commplax.module.core import Signal

class parameters:
    """
    Basic class to be used as a struct of parameters
    """
    pass


@struct.dataclass
class MySignal:
    val: Array                                      # value. [Nfft,Nmodes]
    Fs: float=struct.field(pytree_node=False)       # sampling rate  [Hz]
    sps: int=struct.field(pytree_node=False)        # samples per symbol
    Nch: int=struct.field(pytree_node=False)        # number of channels
    freqspace: float=struct.field(pytree_node=False)# frequency space

    def __add__(self,other):
        return  MySignal(val=self.val + other.val, sps=self.sps, Fs=self.Fs, Nch=self.Nch, freqspace=self.freqspace)
    
    def __sub__(self,other):
        return  MySignal(val=self.val - other.val, sps=self.sps, Fs=self.Fs, Nch=self.Nch, freqspace=self.freqspace)


    def __mul__(self,other):
        return  MySignal(val=self.val * other.val, sps=self.sps, Fs=self.Fs, Nch=self.Nch, freqspace=self.freqspace)

    def __truediv__(self,other):
        return  MySignal(val=self.val / other.val, sps=self.sps, Fs=self.Fs, Nch=self.Nch, freqspace=self.freqspace)


def get_MySignal(val, a):
    return MySignal(val=val, sps=a['sps'], Fs=a['samplerate'], Nch=a['channels'], freqspace=a['freqspace'])


def define_signal(x:Array, E:MySignal, device:str='cpu') -> MySignal:
    '''
        define a signal with device value.
    '''
    if device=='cpu':
        return MySignal(val=jax.device_get(x), sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)
    else:
        return MySignal(val=jax.device_put(x), sps=E.sps, Fs=E.Fs, Nch=E.Nch, freqspace=E.freqspace)



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

