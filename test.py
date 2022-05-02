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
from optical_flax.core import parameters
from optical_flax.dsp import simple_dsp, mimo_dsp
from collections import namedtuple
import pickle
from functools import partial

from gdbp import gdbp_base as gb, data as gdat, aux
import optax
from optical_flax.utils import auto_rho
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

## 测试进度条
from tqdm import tqdm
from time import sleep
for i in tqdm(range(20)):
    sleep(.01)

## Transmitter
from optical_flax.tx import QAM
param = parameters()
param.M   = 16           # modulation formate
param.Rs  = 36e9         # symbol rate [baud]
param.SpS = 16            # samples/symb
param.Nbits = 40000      # number of bits
param.pulse_type = 'rc'   # formato de pulso
param.Ntaps = 4096       # número de coeficientes do filtro RRC
param.alphaRRC = 0.1    # rolloff do filtro RRC
param.Pch_dBm = 0        # potência média por canal WDM [dBm]
param.Nch     = 7       # número de canais WDM
param.Fc      = 299792458/1550E-9 # frequência central do espectro WDM
param.freqSpac = 50e9    # espaçamento em frequência da grade de canais WDM
param.Nmodes = 2         # número de modos de polarização
param.mod = QAM(M=param.M)  # modulation
param.equation = 'NLSE'

# load data
key = jax.random.PRNGKey(0)
key_full = jax.random.split(key, 5)
my_jit = partial(jax.jit, static_argnums=(1,))
sigWDM_Tx, symbTx_ = jax.vmap(simpleWDMTx, in_axes=(0,None), out_axes=0)(key_full, param)
# sigWDM_Tx, symbTx_ = simpleWDMTx(key, param)
print(f'signal shape: {sigWDM_Tx.shape}, symb shape: {symbTx_.shape}')

## channel
np.random.seed(2333)
linearChannel = False
paramCh = parameters()
paramCh.Ltotal = 1125   # km
paramCh.Lspan  = 75     # km
paramCh.alpha = 0.2    # dB/km
paramCh.D = 16.5       # ps/nm/km
paramCh.Fc = 299792458/1550E-9 # Hz
paramCh.hz =  15      # km
paramCh.gamma = 1.3174420805376552    # 1/(W.km)
paramCh.amp = 'edfa'
if linearChannel:
    paramCh.hz = paramCh.Lspan  # km
    paramCh.gamma = 0   # 1/(W.km)
Fs = param.Rs*param.SpS  # sample rates
from optical_flax.models import ssfm, manakov_ssf
sigWDM_ = jax.vmap(manakov_ssf,in_axes=(0,None,None), out_axes=0)(sigWDM_Tx, Fs, paramCh) 
print(sigWDM_.shape)

### Receiver
from optical_flax.rx import simpleRx, sml_dataset
np.random.seed(123)
paramRx = parameters()
paramRx.chid = int(param.Nch / 2)
paramRx.sps = 2
paramRx.lw = 100e3       # linewidth
paramRx.Rs = param.Rs
FO = 64e6           # frequency offset

paramRx.tx_sps = param.SpS
paramRx.pulse = param.pulse
paramRx.Ta = 1/(param.SpS*param.Rs)


key_rx = jax.random.PRNGKey(233)
sigRx = jax.vmap(simpleRx, in_axes=(None,None,None,0,None),out_axes=0)(key, FO, param.freqGrid[paramRx.chid], sigWDM_, paramRx)