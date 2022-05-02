import numpy as np
import jax.numpy as jnp
import jax
from optical_flax.core import parameters
from optical_flax.tx import local_oscillator
from optical_flax.models import coherentReceiver
from optical_flax.dsp import firFilter
import matplotlib.pyplot as plt
from collections import namedtuple
import pickle

Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

def simpleRx(key, FO, freq, sigWDM, paramRx):
    paramRx.N = len(sigWDM)
    paramRx.sps = getattr(paramRx, 'sps', 2)
    paramRx.Fa = paramRx.Rs * paramRx.sps

    sigLO = local_oscillator(key, FO, freq, paramRx)

    ## step 1: coherent receiver
    sigRx1 = jax.vmap(coherentReceiver, in_axes=(-1,None), out_axes=-1)(sigWDM, sigLO)

    ## step 2: match filtering  
    sigRx2 = sigWDM * 0    
    sigRx2 = jax.vmap(firFilter, in_axes=(None, -1), out_axes=-1)(paramRx.pulse, sigRx1)    


    ## step 3: resampling  ## 这一步降采样应该可以优化 ！！
    down_sample_rate = paramRx.tx_sps // paramRx.sps
    sigRx3 = sigRx2[::down_sample_rate, :]
    return sigRx3

def sml_dataset(sigRx, symbTx_, param, paramCh, paramRx, save=True, path='sml_data/dataset'):
    a = {'baudrate': param.Rs,
    'channelindex': paramRx.chid,
    'channels': param.Nch,
    'distance': paramCh.Ltotal * 1e3,
    'lpdbm': 0.0,
    'lpw': 0.001,
    'modformat': '16QAM',
    'polmux': 1,
    'samplerate': param.Rs * paramRx.sps,
    'spans': int(paramCh.Ltotal / paramCh.Lspan),
    'srcid': 'src1',
    'D': paramCh.D, 
    'Fc': 299792458/1550E-9,
    'sps': paramRx.sps, 
    'M':16,
    'CD': 18.451}

    
    symbTx = symbTx_[:,:,paramRx.chid]
    data_train_sml = Input(sigRx, symbTx, 2 * np.pi * paramRx.FO / param.Rs, a)
    if save:
        with open(path,'wb') as file:
            b = pickle.dump((sigRx, symbTx, 2 * np.pi * paramRx.FO / param.Rs, a), file)
        print('data has been saved!')
    return data_train_sml




