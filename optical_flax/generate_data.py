from re import S
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import device_put, device_get
import jax.random as random
from commplax.module import core

import optical_flax.base as base
from optical_flax.models import manakov_ssf, cssfm
from optical_flax.tx import simpleWDMTx
from optical_flax.rx import simpleRx, sml_dataset
from optical_flax.core import parameters
from optical_flax.dsp import simple_dsp, mimo_dsp
from collections import namedtuple
import pickle

from gdbp import gdbp_base as gb, data as gdat, aux
import optax
from optical_flax.utils import auto_rho
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

## 测试进度条
from tqdm import tqdm
from time import sleep



def Tx_data(key, batch, Nch, SpS, Power, Nbits=400000, path = 'data/sml_data/Tx_ch7_N4e6', save=True):
    '''
    Generate Tx data!
    '''
    ## Transmitter
    from commpy.modulation import QAMModem
    param = parameters()
    param.Nch     = Nch       # número de canais WDM

    param.Nbits = Nbits     # number of bits
    param.M   = 16           # modulation formate
    param.Rs  = 36e9         # symbol rate [baud]
    param.SpS = SpS            # samples/symb
    param.pulse_type = 'rc'   # formato de pulso
    param.Ntaps = 4096       # número de coeficientes do filtro RRC
    param.alphaRRC = 0.1    # rolloff do filtro RRC
    param.Pch_dBm = Power    # potência média por canal WDM [dBm]

    param.Fc      = 299792458/1550E-9 # frequência central do espectro WDM
    param.freqSpac = 50e9    # espaçamento em frequência da grade de canais WDM
    param.Nmodes = 2         # número de modos de polarização
    param.mod = QAMModem(m=param.M)  # modulation
    param.equation = 'NLSE'

    # load data
    print('Transmitter is working..')
    key_full = jax.random.split(key, batch)
    sigWDM_Tx, symbTx_ = jax.vmap(simpleWDMTx, in_axes=(0,None), out_axes=0)(key_full, param)
    print(f'signal shape: {sigWDM_Tx.shape}, symb shape: {symbTx_.shape}')

    if save:
        with open(path,'wb') as file:
            pickle.dump((sigWDM_Tx, symbTx_, param), file)
        print('Data has been saved!')

    return 0

def Rx_data(key, tx_data_path, rx_data_path='sml_data/dataset_ch7_N4e6_dz1.5', dz=1.5, sps=2):
    '''
    generate Rx data
    '''

    # step 1: load Tx data
    with open(tx_data_path,'rb') as file:
        sigWDM_Tx, symbTx_, param = pickle.load(file)
    print('Tx data has been loaded in')
    print(f'signal shape: {sigWDM_Tx.shape}, symb shape: {symbTx_.shape}')
    

    # STEP 2: fiber transmitting
    print('channel working!')
    linearChannel = False
    paramCh = parameters()
    paramCh.Ltotal = 1125   # km
    paramCh.Lspan  = 75     # km
    paramCh.alpha = 0.2    # dB/km
    paramCh.D = 16.5       # ps/nm/km
    paramCh.Fc = 299792458/1550E-9 # Hz
    paramCh.hz =  dz      # km
    paramCh.gamma = 1.3174420805376552    # 1/(W.km)
    paramCh.amp = 'edfa'
    if linearChannel:
        paramCh.hz = paramCh.Lspan  # km
        paramCh.gamma = 0   # 1/(W.km)
    Fs = param.Rs*param.SpS  # sample rates

    if len(sigWDM_Tx.shape) == 3:
        sigWDM = jax.vmap(manakov_ssf,in_axes=(0,None,None), out_axes=0)(sigWDM_Tx, Fs, paramCh) 
    else:
        sigWDM = manakov_ssf(sigWDM_Tx, Fs, paramCh) 

    print('channel transmission done!')
    print('Receiver is working...')

    # step 3: Rx
    paramRx = parameters()
    paramRx.chid = int(param.Nch / 2)
    paramRx.sps = sps
    FO = 64e6*3
    paramRx.FO = FO         # frequency offset
    paramRx.lw = 100e3          # linewidth
    paramRx.Rs = param.Rs
    paramRx.tx_sps = param.SpS
    paramRx.pulse = param.pulse
    paramRx.freq = param.freqGrid[paramRx.chid]
    paramRx.Ta = 1/(param.SpS*param.Rs)

    if len(sigWDM.shape) == 3:
        key_full = jax.random.split(key, sigWDM.shape[0])
        sigRx = jax.vmap(simpleRx, in_axes=(0,None,None,0,None),out_axes=0)(key_full, FO, param.freqGrid[paramRx.chid], sigWDM, paramRx)
    else:
        sigRx = simpleRx(key, FO, param.freqGrid[paramRx.chid], sigWDM, paramRx)
    data_sml = sml_dataset(sigRx, symbTx_, param, paramCh, paramRx, save=True, path=rx_data_path)
    return 0

from collections import namedtuple
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

def get_data(path,sps=2, batch=False):
    '''
        down sampling algrithm
    '''
    with open(path,'rb') as file:
        b = pickle.load(file)  # b = (y,x,w0,a)
        a = b[3]
        a['samplerate'] = a['baudrate'] * sps 
        w0 = -b[2]
    if len(b[0].shape) == 3:
        if batch:
            y = b[0][:,::(a['sps'] // sps),:]
            x = b[1]
        else:
            y = b[0][0,::(a['sps'] // sps),:]
            x = b[1][0]
    else:
        y = b[0][::(a['sps'] // sps)]
        x = b[1]
    
    a['sps'] = sps
    return Input(y, x, w0, a)



