import numpy as np
import jax.numpy as jnp
import jax
from commpy.modulation import QAMModem
from optical_flax.operator import circFilter, frame
from commpy.filters import rrcosfilter, rcosfilter
from functools import partial

def mzm(Ai, Vπ, u, Vb):
    """
    MZM modulator 
    
    :param Vπ: Vπ-voltage
    :param Vb: bias voltage
    :param u:  modulator's driving signal (real-valued)
    :param Ai: amplitude of the input CW carrier
    
    :return Ao: output optical signal
    """
    π  = np.pi
    Ao = Ai*jnp.cos(0.5/Vπ*(u+Vb)*π)
    
    return Ao


def iqm(Ai, u, Vπ, VbI, VbQ):
    """
    IQ modulator 
    
    :param Vπ: MZM Vπ-voltage
    :param VbI: in-phase MZM bias voltage
    :param VbQ: quadrature MZM bias voltage    
    :param u:  modulator's driving signal (complex-valued baseband)
    :param Ai: amplitude of the input CW carrier
    
    :return Ao: output optical signal
    """
    Ao = mzm(Ai/jnp.sqrt(2), Vπ, u.real, VbI) + 1j*mzm(Ai/jnp.sqrt(2), Vπ, u.imag, VbQ)
    
    return Ao

def signal_power(signal):
    """
    Compute the power of a discrete time signal.

    Parameters
    ----------
    signal : 1D ndarray
             Input signal.

    Returns
    -------
    P : float
        Power of the input signal.
    """
    return jnp.mean(jnp.abs(signal)**2)

def pulseShape(pulseType, SpS=2, N=1024, alpha=0.1, Ts=1):
    """
    Generate pulse shaping filters
    
    :param pulseType: 'rect','nrz','rrc'
    :param SpS: samples per symbol
    :param N: number of filter coefficients
    :param alpha: RRC rolloff factor
    :param Ts: symbol period
    
    :return filterCoeffs: normalized filter coefficients   
    """  
    fa = (1/Ts)*SpS    # sample rate
    
    t = np.linspace(-2, 2, SpS)
    Te = 1       
    
    if pulseType == 'rect':
        filterCoeffs = np.concatenate((np.zeros(int(SpS/2)), np.ones(SpS), np.zeros(int(SpS/2))))       
    elif pulseType == 'nrz':
        filterCoeffs = np.convolve(np.ones(SpS), 2/(np.sqrt(np.pi)*Te)*np.exp(-t**2/Te), mode='full')        
    elif pulseType == 'rrc':
        tindex, filterCoeffs = rrcosfilter(N, alpha, Ts, fa)
    elif pulseType == 'rc':
        tindex, filterCoeffs = rcosfilter(N, alpha, Ts, fa)
        
    return filterCoeffs/np.sqrt(np.sum(filterCoeffs**2))

class QAM(object):
    '''
    QAM Modulation format.
    '''
    def __init__(self, M,reorder_as_gray=True):
        self.M = M
        num_symb_pam = np.sqrt(M)
        if num_symb_pam != int(num_symb_pam):
            raise ValueError('m must lead to a square QAM.')
        pam = np.arange(-num_symb_pam + 1, num_symb_pam, 2)
        constellation = np.tile(np.hstack((pam, pam[::-1])), int(num_symb_pam) // 2) * 1j + pam.repeat(num_symb_pam)
        if reorder_as_gray:
            m = np.log2(self.M)
            from sympy.combinatorics.graycode import GrayCode
            gray_code_sequence = GrayCode(m).generate_gray()
            gray_code_sequence_array = np.fromiter((int(g, 2) for g in gray_code_sequence), int, len(constellation))
            self.constellation = jnp.array(constellation)[gray_code_sequence_array.argsort()]
        else:
            self.constellation = jnp.array(constellation)
        
        self.num_bits_symbol = int(np.log2(self.M))
        self.Es = signal_power(self.constellation)

    
    def bit2symbol(self, bits):
        N = len(bits)
        pow = 2**jnp.arange(N-1,-1,-1)
        idx = jnp.sum(pow*bits)
        return self.constellation[idx]


    def modulate(self, bits):
        bits_batch = frame(bits, self.num_bits_symbol, self.num_bits_symbol)
        symbol_batch = jax.vmap(self.bit2symbol)(bits_batch)
        return symbol_batch
    
    def demodulate(self,symb):
        return symb


@partial(jax.jit, static_argnums=1)
def upsample(x, n):
    """
    Upsample the input array by a factor of n

    Adds n-1 zeros between consecutive samples of x

    Parameters
    ----------
    x : 1D ndarray
        Input array.

    n : int
        Upsampling factor

    Returns
    -------
    y : 1D ndarray
        Output upsampled array.
    """
    y = jnp.empty(len(x) * n, dtype=jnp.complex64)
    y = y.at[0::n].set(x)
    return y


def phaseNoise(key, lw, Nsamples, Ts):
    
    σ2 = 2*np.pi*lw*Ts    
    phi = jax.random.normal(key,(Nsamples,),jnp.float32) * jnp.sqrt(σ2)
  
    return jnp.cumsum(phi)


#@partial(jax.jit, static_argnums=(3,4))
def local_oscillator(key, FO, freq, N, paramLo):
          
    Plo     = 10**(paramLo.Plo_dBm/10)*1e-3            # power in W
    Δf_lo   = freq + FO              # downshift of the channel to be demodulated 
                                    
    # generate LO field
    π       = jnp.pi
    t       = jnp.arange(0, N) * paramLo.Ta
    ϕ_pn_lo = phaseNoise(key, paramLo.lw, N, paramLo.Ta)    # gaussian process
    sigLO   = jnp.sqrt(Plo) * jnp.exp(1j*(2*π*Δf_lo*t + paramLo.ϕ_lo + ϕ_pn_lo))

    return sigLO, ϕ_pn_lo



def wdm_base(Nfft, fa, freqGrid):
    t = jnp.arange(0, Nfft)
    wdm_wave = jnp.exp(1j*2*jnp.pi/fa * freqGrid[None,:]*t[:,None]) # [Nsymb*SpS, Nch]
    return wdm_wave



@partial(jax.jit,static_argnums=(3,))
def simpleWDMTx(key, pulse, wdm_wave, param):
    """
    Simple WDM transmitter
    
    Generates a complex baseband waveform representing a WDM signal with arbitrary number of carriers
    
    :param.M: QAM order [default: 16]
    :param.Rs: carrier baud rate [baud][default: 32e9]
    :param.SpS: samples per symbol [default: 16]
    :param.Nbits: total number of bits per carrier [default: 60000]
    :param.pulse: pulse shape ['nrz', 'rrc'][default: 'rrc']
    :param.Ntaps: number of coefficients of the rrc filter [default: 4096]
    :param.alphaRRC: rolloff do rrc filter [default: 0.01]
    :param.Pch_dBm: launched power per WDM channel [dBm][default:-3 dBm]
    :param.Nch: number of WDM channels [default: 5]
    :param.Fc: central frequency of the WDM spectrum [Hz][default: 193.1e12 Hz]
    :param.freqSpac: frequency spacing of the WDM grid [Hz][default: 40e9 Hz]
    :param.Nmodes: number of polarization modes [default: 1]

    """


    # transmitter parameters
    Ts  = 1/param.Rs        # symbol period [s]
    Fa  = 1/(Ts/param.SpS)  # sampling frequency [samples/s]
    Ta  = 1/Fa              # sampling period [s]
    Pch = 10**(param.Pch_dBm/10)*1e-3   # optical signal power per WDM channel [W]


    # modulation scheme
    mod = QAM(M=param.M)
    Es = mod.Es

    vmap = partial(jax.vmap, in_axes=(-1, None), out_axes=-1)


    def one_channel(key, pulse):
        # step 1: generate random bits      bitsTx: [Nbits,]
        bitsTx = jax.random.randint(key, (param.Nbits,), 0, 2)
        # step 2: map bits to constellation symbols  symbTx: [Nsymb,]
        symbTx = mod.modulate(bitsTx)
        # step 3: normalize symbols energy to 1
        symbTx = symbTx/jnp.sqrt(Es)
        # step 4: upsampling   symbolsUp :  [Nsymb*SpS] d
        symbolsUp = upsample(symbTx, param.SpS)
        # step 5: pulse shaping
        sigTx = circFilter(pulse, symbolsUp)
        # step 6: optical modulation
        sigTxCh = iqm(param.Ai, 0.5*sigTx, param.Vπ, param.Vb, param.Vb)
        # step 7: set the tx power [sqrt(W)] :   Pch [W] each channel
        sigTxCh = jnp.sqrt(Pch/param.Nmodes) * sigTxCh / jnp.sqrt(signal_power(sigTxCh))
        return sigTxCh, symbTx
    
    key_full = jax.random.split(key, param.Nch*param.Nmodes).reshape(2, param.Nch, param.Nmodes)
    sigWDM, SymbTx = vmap(vmap(one_channel))(key_full, pulse) #[Nsymb*SpS, Nch, Nmodes]


    if param.equation == 'NLSE':
        sigTxWDM = jnp.sum(wdm_wave[...,None]*sigWDM, axis=-2) # [Nsymb*SpS, Nmodes]
    elif param.equation == 'WDM-NLSE':
        sigTxWDM = sigWDM
    else:
        raise(ValueError)
       
    return sigTxWDM, SymbTx