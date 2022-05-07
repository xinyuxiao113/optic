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
from optical_flax.dsp import simple_dsp, mimo_dsp, pulseShape
from collections import namedtuple
import pickle

from gdbp import gdbp_base as gb, data as gdat, aux
import optax
from optical_flax.utils import auto_rho
DataInput = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

## 测试进度条
from tqdm import tqdm
from time import sleep



def Tx_data(key, batch, Nch, SpS=16, Power=0, Nbits=400000):
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

    # pulse shaping
    Ts  = 1/param.Rs        # symbol period [s]
    if param.pulse_type == 'nrz':
        pulse = pulseShape('nrz', param.SpS)
    elif param.pulse_type == 'rrc':
        pulse = pulseShape('rrc', param.SpS, N=param.Ntaps, alpha=param.alphaRRC, Ts=Ts)
    elif param.pulse_type == 'rc':
        pulse = pulseShape('rc', param.SpS, N=param.Ntaps, alpha=param.alphaRRC, Ts=Ts)
    pulse = jax.device_put(pulse/np.max(np.abs(pulse)))
    param.pulse = pulse

    # central frequencies of the WDM channels
    freqGrid = np.arange(-int(param.Nch/2), int(param.Nch/2)+1,1)*param.freqSpac
    if (param.Nch % 2) == 0:
        freqGrid += param.freqSpac/2

    if param.equation == 'NLSE':
        param.freqGrid = freqGrid
    else:
        param.freqGrid = freqGrid*0

    # load data
    print('Transmitter is working..')
    key_full = jax.random.split(key, batch)
    signal = []
    symbol = []
    for i in range(batch):
        sigWDM_Tx, symbTx_ = simpleWDMTx(key_full[i], pulse, param)
        signal.append(sigWDM_Tx)
        symbol.append(symbTx_)

    sigWDM = jnp.stack(signal, 0)
    symbWDM = jnp.stack(symbol, 0)
    print(f'signal shape: {sigWDM.shape}, symb shape: {symbWDM.shape}')

    return sigWDM, symbWDM, param


def channel(sigWDM_Tx, Fs, dz=1):
    print('data transmition...')
    np.random.seed(2333)
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
    paramCh.NF = 4.5
    if linearChannel:
        paramCh.hz = paramCh.Lspan  # km
        paramCh.gamma = 0   # 1/(W.km)
    
    if len(sigWDM_Tx.shape) == 3:
        sigWDM = jax.vmap(manakov_ssf,in_axes=(0,None,None), out_axes=0)(sigWDM_Tx, Fs, paramCh) 
    else:
        sigWDM = manakov_ssf(sigWDM_Tx, Fs, paramCh) 
    print('channel transmission done!')
    print(f'Signal shape {sigWDM.shape}')
    return sigWDM, paramCh



def Rx_data(key, sigWDM, symbWDM, sps, param, paramCh):
    '''
    generate Rx data
    '''
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
    data_sml = sml_dataset(sigRx, symbWDM, param, paramCh, paramRx)
    return data_sml





from collections import namedtuple
DataInput = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

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
    return DataInput(y, x, w0, a)



