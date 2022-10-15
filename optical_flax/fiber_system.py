import numpy as np
import pickle
import jax
import jax.numpy as jnp
from commpy.modulation import QAMModem

from optical_flax.fiber_channel import manakov_ssf
from optical_flax.fiber_tx import simpleWDMTx, pulseShape, wdm_base
from optical_flax.fiber_rx import simpleRx, sml_dataset
from optical_flax.utils import calc_time
from optical_flax.core import parameters
from collections import namedtuple


DataInput = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

from flax import struct
from typing import Any, NamedTuple, Iterable, Callable, Optional
from typing import Callable

Array = Any
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
    


@calc_time
def Tx_data(key, batch, Nch, SpS=32, Power=0, Nbits=400000, Rs=190e9, freq_space=220e9, Nmodes=2, equation='NLSE', M=16):
    '''
    Generate dual-polarization 16-QAM Tx data.
    Nsymb = Nbits / log2(16)

    Input:
        key: rng.
        batch: an integer.The number of batch.
        Nch: an integer. The number of channels.
        SpS: Tx samples per symbol.
        Power: tx power. [dBW]
        Nbits: number of bits for each channel.
        Rs: symb rate.[Hz]
        freq_space: frequence space of two neibough channel.
        Nmodes: number of polarization modes.
        equation: 'NLSE' or 'WDM-NLSE'
    Output:
        sigWDM: a jax array with shape:[batch, Nsamples, pmodes] 
                repesents the signal for transmission.
                Nsamples = Nsymb * SpS, Nsymb = Nbits/log2(16), pmodes = 2.
        symbWDM: a jax array with shape:[batch, Nsymb, Nch, pmodes]
                represents the symbols for transmission.
        param: a parameter structure. Reccording information of transmitter.
    
    More inforamtion:
        [C band] transmission windows. 1530[nm] to 1565[nm]
        central wavelength: 1550 [nm]
        See https://en.wikipedia.org/wiki/Fiber-optic_communication#cite_note-64 for more windows.

    '''
    # Setting transmitter parameters
    param = parameters()
    param.Pch_dBm = Power    # Power[dBm]
    param.Nch     = Nch      # número de canais WDM
    param.Nbits = Nbits     # number of bits
    param.freqSpac = freq_space   # espaçamento em frequência da grade de canais WDM
    param.Nmodes = Nmodes         # número de modos de polarização
    param.Rs  = Rs         # symbol rate [baud]
    param.SpS = SpS            # samples/symb
    param.equation = equation
    param.M   = M           # modulation formate


    # fixed parameters.
    param.pulse_type = 'rc'   # formato de pulso
    param.Ntaps = 4096       # número de coeficientes do filtro RRC
    param.alphaRRC = 0.1    # rolloff do filtro RRC
    param.Fc      = 299792458/1550E-9 # frequência central do espectro WDM
    param.mod = QAMModem(m=param.M)  # modulation
    # setting IQM param
    param.Ai = 1
    param.Vπ = 2
    param.Vb = -2



    # Verify sampling theorem
    fa = param.Rs * param.SpS
    fc = param.Nch / 2 * param.freqSpac
    print('Sample rate fa: %g, Cut off frequency fc: %g, fa > 2fc: %s' % (fa, fc, fa> 2*fc))
    if fa < 2*fc:
        print('sampling thm does not hold!')
        raise(ValueError)

    # Pulse generation
    Ts  = 1/param.Rs        # symbol period [s]
    if param.pulse_type == 'nrz':
        pulse = pulseShape('nrz', param.SpS)
    elif param.pulse_type == 'rrc':
        pulse = pulseShape('rrc', param.SpS, N=param.Ntaps, alpha=param.alphaRRC, Ts=Ts)
    elif param.pulse_type == 'rc':
        pulse = pulseShape('rc', param.SpS, N=param.Ntaps, alpha=param.alphaRRC, Ts=Ts)
    pulse = jax.device_put(pulse/np.max(np.abs(pulse)))
    param.pulse = pulse

    # WDM waves generation
    freqGrid = jnp.arange(-int(param.Nch/2), int(param.Nch/2)+1,1)*param.freqSpac
    if (param.Nch % 2) == 0:
        freqGrid += param.freqSpac/2
    
    param.freqGrid = freqGrid

    Nfft = ((param.Nbits)//np.log2(param.M))*param.SpS
    wdm_wave = wdm_base(Nfft,fa,param.freqGrid)# [Nsymb*SpS, Nch]

    # load data
    print('Transmitter is working..')
    key_full = jax.random.split(key, batch)
    signal = []
    symbol = []
    
    for i in range(batch):
        sigWDM_Tx, symbTx_ = simpleWDMTx(key_full[i], pulse, wdm_wave,  param)
        signal.append(jax.device_get(sigWDM_Tx))
        symbol.append(jax.device_get(symbTx_))

    sigWDM = np.stack(signal, 0)
    symbWDM = np.stack(symbol, 0)
   
    if batch == 1:
        sigWDM = sigWDM[0]
        symbWDM = symbWDM[0]
    
    print(f'signal shape: {sigWDM.shape}, symb shape: {symbWDM.shape}')
    return sigWDM, symbWDM, param


@calc_time
def channel(key, sigWDM_Tx, Fs, dz=1, module=manakov_ssf):
    '''
    optical fiber channel model.
    Input:
        key: rng for channel EDFA noise.
        sigWDM_Tx: a jax array with shape [batch, Nsamples, pmodes]
                signal before optical fiber transmission.
        Fs: a integer. Sample rates of sigWDM_Tx. 
            Fs = Symbrates * SpS
        dz: SSFM step size. [km]
    
    Output:
        sigWDM:
            signal after optical fiber transmission.
        paramCh:
            a parameter structure. Reccording information of channels.
        
    '''
    print('data transmition...')
    sigWDM_Tx = jax.device_put(sigWDM_Tx)
    linearChannel = False
    paramCh = parameters()
    paramCh.Ltotal = 2000   # km
    paramCh.Lspan  = 80     # km
    paramCh.alpha = 0.2    # dB/km
    paramCh.D = 16.5       # ps/nm/km
    paramCh.Fc = 299792458/1550E-9 # Hz
    paramCh.hz =  dz      # km
    paramCh.gamma = 1.6567    # 1/(W.km)
    paramCh.amp = 'edfa'
    paramCh.NF = 4.5
    if linearChannel:
        paramCh.hz = paramCh.Lspan  # km
        paramCh.gamma = 0   # 1/(W.km)
    
    if len(sigWDM_Tx.shape) == 3:
        key_full = jax.random.split(key, sigWDM_Tx.shape[0])
        sigWDM = jax.vmap(module,in_axes=(0, 0,None,None), out_axes=0)(key_full, sigWDM_Tx, Fs, paramCh) 
    else:
        sigWDM = module(key, sigWDM_Tx, Fs, paramCh) 
    print('channel transmission done!')
    print(f'Signal shape {sigWDM.shape}')
    return jax.device_get(sigWDM), paramCh


@calc_time
def Rx_data(key, sigWDM, symbWDM, sps, param, paramCh, FO=64e6, lw=100e3, setting='ideal'):
    '''
    generate Central channel rx data.

    Input: 
        sigWDM: [batch, Nsample, Nmodes]
        symbWDM: [batch, Nsymb, channels, Nmodes]
        sps: int = 2, rx sps
        param: prameters of Tx
        FO: frequency offset
        lw: linewidth of noise
        paramCh: prameters of Fiber Channel
        setting: 'simpleRx' or 'ideal'
    Output:
        data_sml:
            dataset, a namedtuple('DataInput', ['y', 'x', 'w0', 'a'])
            y: jax array. [batch, Nsamples, pmodes]
               received signal for central channel.
            x: jax array. [batch, Nsymbol, pmodes] 
               truth symbol for central channel.
            w0: estimated FO(frequency offset).
                w0 \approx 2*pi*FO/Rs
            a: a dict. additional information. A example:
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
    '''
    paramRx = parameters()
    # TODO: change the channel id.
    paramRx.chid = int(param.Nch / 2)
    paramRx.sps = sps
    paramRx.FO = FO         # frequency offset
    paramRx.lw = lw          # linewidth
    paramRx.Rs = param.Rs
    paramRx.tx_sps = param.SpS
    paramRx.pulse = param.pulse
    paramRx.freq = param.freqGrid[paramRx.chid]
    paramRx.Ta = 1/(param.SpS*param.Rs)    # 发射信号的样本时间
    paramRx.Fa = paramRx.Rs * paramRx.sps  # 接收机处的 采样率

    paramRx.Plo_dBm = 10         # Local occilator power in dBm
    paramRx.ϕ_lo = 0.0           # initial phase in rad  

    if setting == 'simpleRx':
        if len(sigWDM.shape) == 3:
            key_full = jax.random.split(key, sigWDM.shape[0])
            sigRx, noise = jax.vmap(simpleRx, in_axes=(0,None,None,0,None),out_axes=0)(key_full, FO, param.freqGrid[paramRx.chid], sigWDM, paramRx)
        else:
            sigRx, noise = simpleRx(key, FO, param.freqGrid[paramRx.chid], sigWDM, paramRx)
    elif setting=='ideal':
        if len(sigWDM.shape) == 3:
            key_full = jax.random.split(key, sigWDM.shape[0])
            sigRx, noise = jax.vmap(simpleRx, in_axes=(0,None,None,0,None),out_axes=0)(key_full, FO, param.freqGrid[paramRx.chid], sigWDM, paramRx)
        else:
            sigRx, noise = simpleRx(key, FO, param.freqGrid[paramRx.chid], sigWDM, paramRx)

    data_sml, paramRx = sml_dataset(sigRx, symbWDM, param, paramCh, paramRx)
    return data_sml, paramRx, noise





DataInput = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

def get_data(path,sps=2, batch=False, opposite_sign=False):
    '''
        TODO: more down sampling algrithm

        Input:
            path: dataset path.
            sps: rx sps.
            batch: True -> y,x has a batch axis.  False --> y,x has no batch axis.
            opposite_sign: reverse the sign of w0 or not.
        Output:
            dataset:(y,x,w0,a)
    '''
    with open(path,'rb') as file:
        b,paramRx,noise = pickle.load(file)  # b = (y,x,w0,a)
        a = b[3]
        a['samplerate'] = a['baudrate'] * sps 
        if opposite_sign:
            w0 = -b[2]
        else:
            w0 = b[2]
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



