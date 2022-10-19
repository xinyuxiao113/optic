import numpy as np
import jax.numpy as jnp
import jax
import scipy.constants as const
from collections import namedtuple
from jax.numpy.fft import fft, ifft, fftfreq

from optical_flax.fiber_tx import local_oscillator, phaseNoise
from optical_flax.operator import firFilter, circFilter, L2
from optical_flax.core import MySignal, get_omega

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
    
    Ei = jnp.array([E1, jnp.zeros((E1.size,)), jnp.zeros((E1.size,)), E2])    # [4, N]
    
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

    ## step 0: WDM split


    ## step 1: coherent receiver
    sigRx1 = jax.vmap(coherentReceiver, in_axes=(-1,None), out_axes=-1)(sigWDM, sigLO)

    ## step 2: match filtering  
    # sigRx2 = jax.vmap(circFilter, in_axes=(None, -1), out_axes=-1)(paramRx.pulse, sigRx1)  
    # down_sample_rate = paramRx.tx_sps // paramRx.sps
    # sigRx2 = sigRx2[::down_sample_rate, :]

    E = MySignal(val=sigRx1, Fs=1/paramRx.Ta, sps=paramRx.tx_sps, Nch=paramRx.Nch, freqspace=paramRx.freqspace)  
    sigRx2 = rx(E, paramRx.chid, paramRx.sps).val

    ## step 3: resampling # TODO: 可以优化！
    sigRx = sigRx2/L2(sigRx2)
    return sigRx, ϕ_pn_lo


def idealRx(key, E, chid, rx_sps, FO=0, lw=0, R=1, Plo=10):
    '''
    Input:
        key: rng for rx noise.
        E: WDM signal.  E.val [N, Nmodes]
        chid: channel id from [0,1,2,...,Nch-1].
        rx_sps: output sps.
        FO: float. frequency offset. [Hz]
        lw: linewidth of LO.  
    Output:
        sigRx: [Nsymb * rx_sps, pmodes]
        phi_pn: [Nsamples] noise.

    '''
    N = E.val.shape[0]

    # sigLO, ϕ_pn_lo = local_oscillator(key, FO, freq, N, paramRx)

    ## step 0: WDM split
    E1 = rx(E, chid, rx_sps)  

    ## step 1: phase noise
    N1 = E1.val.shape[0]
    t  = jnp.arange(0, N1) * 1 / E1.Fs
    phi = 2*np.pi*FO*t +  phaseNoise(key, lw, N1, 1/E1.Fs)
    y = R*jnp.exp(Plo) * E1.val * jnp.exp(-1j*phi[:,None]) 
    y = y / L2(y)
    return y, phi



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
    'distance': paramCh.Ltotal * 1e3,    #【m】
    'lpdbm': param.Pch_dBm,    # [dBm]
    'lpw': 10**(param.Pch_dBm/10)*1e-3, # [W]
    'modformat': f'{param.M}QAM',
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
    'CD': 18.451,
    'freqspace': param.freqSpac}

    if symbTx_.ndim == 4:
        symbTx = symbTx_[:,:,paramRx.chid]
    elif symbTx_.ndim==3:
        symbTx = symbTx_[:,paramRx.chid]
    else:
        raise(ValueError)

    # FO这里取了负号，get_data就不用了
    w0 = - 2 * np.pi * paramRx.FO / param.Rs  # phase rotation each symbol.
    data_train_sml = DataInput(sigRx, symbTx,w0, a)
    return data_train_sml, paramRx



def rx(E: MySignal, chid:int, new_sps:int) -> MySignal:
    ''' 
    Get single channel information from WDM signal.
    Input:
        E: 1D array. WDM signal. (Nfft,Nmodes)
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


