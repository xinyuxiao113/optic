import numpy as np
import jax.numpy as jnp
import jax
import scipy.constants as const
from optical_flax.fiber_tx import local_oscillator
from optical_flax.operator import firFilter
from collections import namedtuple
from jax.numpy.fft import fft, ifft, fftfreq

DataInput = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

def balancedPD(E1, E2, R=1):
    """
    Balanced photodetector (BPD)
    
    :param E1: input field [nparray]
    :param E2: input field [nparray]
    :param R: photodiode responsivity [A/W][scalar, default: 1 A/W]
    
    :return: balanced photocurrent
    """
    # assert R > 0, 'PD responsivity should be a positive scalar'
    assert E1.size == E2.size, 'E1 and E2 need to have the same size'
    
    i1 = R*E1 * jnp.conj(E1)
    i2 = R*E2 * jnp.conj(E2)    

    return i1-i2

def hybrid_2x4_90deg(E1, E2):
    """
    Optical 2 x 4 90° hybrid
    
    :param E1: input signal field [nparray]
    :param E2: input LO field [nparray]
        
    :return: hybrid outputs
    """
    assert E1.size == E2.size, 'E1 and E2 need to have the same size'
    
    # optical hybrid transfer matrix    
    T = jnp.array([[ 1/2,  1j/2,  1j/2, -1/2],
                  [ 1j/2, -1/2,  1/2,  1j/2],
                  [ 1j/2,  1/2, -1j/2, -1/2],
                  [-1/2,  1j/2, -1/2,  1j/2]])
    
    Ei = jnp.array([E1, jnp.zeros((E1.size,)), 
                   jnp.zeros((E1.size,)), E2])    
    
    Eo = T@Ei
    
    return Eo

def coherentReceiver(Es, Elo, Rd=1):
    """
    Single polarization coherent optical front-end
    
    :param Es: input signal field [nparray]
    :param Elo: input LO field [nparray]
    :param Rd: photodiode resposivity [scalar]
    
    :return: downconverted signal after balanced detection    
    """
    assert Es.size == Elo.size, 'Es and Elo need to have the same size'
    
    # optical 2 x 4 90° hybrid 
    Eo = hybrid_2x4_90deg(Es, Elo)
        
    # balanced photodetection
    sI = balancedPD(Eo[1,:], Eo[0,:], Rd)
    sQ = balancedPD(Eo[2,:], Eo[3,:], Rd)
    
    return sI + 1j*sQ


def linFiberCh(Ei, L, alpha, D, Fc, Fs):
    """
    Linear fiber channel w/ loss and chromatic dispersion

    :param Ei: optical signal at the input of the fiber
    :param L: fiber length [km]
    :param alpha: loss coeficient [dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km]   
    :param Fc: carrier frequency [Hz]
    :param Fs: sampling frequency [Hz]
    
    :return Eo: optical signal at the output of the fiber
    """
    #c  = 299792458   # speed of light [m/s](vacuum)    
    c_kms = const.c/1e3
    λ  = c_kms/Fc
    α  = alpha/(10*np.log10(np.exp(1)))
    β2 = -(D*λ**2)/(2*np.pi*c_kms)
    
    Nfft = len(Ei)

    ω = 2*np.pi*Fs*fftfreq(Nfft)
    ω = ω.reshape(ω.size,1)
    
    try:
        Nmodes = Ei.shape[1]
    except IndexError:
        Nmodes = 1
        Ei = Ei.reshape(Ei.size,Nmodes)

    ω = jnp.tile(ω,(1, Nmodes))
    Eo = ifft(fft(Ei,axis=0) * jnp.exp(-α*L - 1j*(β2/2)*(ω**2)*L), axis=0)
    
    if Nmodes == 1:
        Eo = Eo.reshape(Eo.size,)
        
        
    return Eo, jnp.exp(-α*L - 1j*(β2/2)*(ω**2)*L)

def simpleRx(key, FO, freq, sigWDM, paramRx):
    '''
    Input:
        key: rng for rx noise.
        FO: float. frequency offset.
        freq: float. frequency for our interested channel.
        sigWDM: a jax array with shape [Nsamples,pmodes]
        paramRx: parameters for rx.
    Output:
        sigRx3: [Nsymb * rx_sps, pmodes]
        ϕ_pn_lo: [Nsamples] noise.

    '''
    N = len(sigWDM)

    sigLO, ϕ_pn_lo = local_oscillator(key, FO, freq, N, paramRx)

    ## step 1: coherent receiver
    sigRx1 = jax.vmap(coherentReceiver, in_axes=(-1,None), out_axes=-1)(sigWDM, sigLO)
    ## step 2: match filtering  
    sigRx2 = sigWDM * 0    
    sigRx2 = jax.vmap(firFilter, in_axes=(None, -1), out_axes=-1)(paramRx.pulse, sigRx1)    


    ## step 3: resampling # TODO: 可以优化！
    down_sample_rate = paramRx.tx_sps // paramRx.sps
    sigRx3 = sigRx2[::down_sample_rate, :]
    return sigRx3, ϕ_pn_lo

def sml_dataset(sigRx, symbTx_, param, paramCh, paramRx):
    '''
        generate dataset.
        Input:
            sigRx: [batch, N, pmodes] or [N,pmodes]
            symbTx_: The full channel symbols. [batch, Nsymb, channels,pmodes]
            param: Tx param.
            paramCh: channel param.
            paramRx: rx param.
        Output:
            DataInput = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])
    '''
    a = {'baudrate': param.Rs,
    'channelindex': paramRx.chid,
    'channels': param.Nch,
    'distance': paramCh.Ltotal * 1e3,
    'lpdbm': param.Pch_dBm,    # [dBm]
    'lpw': 10**(param.Pch_dBm/10)*1e-3, # [W]
    'modformat': '16QAM',
    'polmux': 1,
    'samplerate': param.Rs * paramRx.sps,
    'spans': int(paramCh.Ltotal / paramCh.Lspan),
    'srcid': 'src1',
    'D': paramCh.D * 1e-6,   #[s/m^2]
    'carrier_frequency': param.Fc + param.freqGrid[paramRx.chid],
    'fiber_loss': paramCh.alpha*1e-3, # [dB/m]
    'gamma': paramCh.gamma * 1e-3,    # [1/W/m]
    'sps': paramRx.sps, 
    'M': param.M,
    'CD': 18.451}

    if symbTx_.ndim == 4:
        symbTx = symbTx_[:,:,paramRx.chid]
    elif symbTx_.ndim==3:
        symbTx = symbTx_[:,paramRx.chid]
    else:
        raise(ValueError)

    # FO这里取了负号，get_data就不用了
    data_train_sml = DataInput(sigRx, symbTx, - 2 * np.pi * paramRx.FO / param.Rs, a)
    return data_train_sml, paramRx




