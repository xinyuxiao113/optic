from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from optical_flax.operator import fft,ifft,fftfreq,fftshift
from numpy.random import normal
import scipy.constants as const
from tqdm import tqdm


def edfa(key, Ei, Fs=None, G=20, NF=4.5, Fc=193.1e12):
    """
    Simple EDFA model

    :param Ei: input signal field [nparray]
    :param Fs: sampling frequency [Hz][scalar]
    :param G: gain [dB][scalar, default: 20 dB]
    :param NF: EDFA noise figure [dB][scalar, default: 4.5 dB]
    :param Fc: optical center frequency [Hz][scalar, default: 193.1e12 Hz]    

    :return: amplified noisy optical signal [nparray]
    """
    # assert G > 0, 'EDFA gain should be a positive scalar'
    # assert NF >= 3, 'The minimal EDFA noise figure is 3 dB'
    
    NF_lin   = 10**(NF/10)
    G_lin    = 10**(G/10)
    nsp      = G_lin*NF_lin / (2*(G_lin - 1))
    N_ase    = (G_lin - 1)*nsp*const.h*Fc
    p_noise  = N_ase*Fs    
    noise    = jax.random.normal(key, Ei.shape, dtype=jnp.complex64) * np.sqrt(p_noise)
    return Ei * np.sqrt(G_lin) + noise


@partial(jax.jit, static_argnums=(2,3,4))
def ssfm(key, Ei, Fs, paramCh, order=2):      
    """
    Split-step Fourier method (symmetric, single-pol.)

    :param Ei: input signal
    :param Fs: sampling frequency of Ei [Hz]
    :param paramCh: object with physical parameters of the optical channel
    
    :paramCh.Ltotal: total fiber length [km][default: 400 km]
    :paramCh.Lspan: span length [km][default: 80 km]
    :paramCh.hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :paramCh.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :paramCh.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :paramCh.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :paramCh.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    :paramCh.amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    :paramCh.NF: edfa noise figure [dB] [default: 4.5 dB]    
    
    :return Ech: propagated signal
    """


    Ltotal = paramCh.Ltotal 
    Lspan  = paramCh.Lspan
    hz     = paramCh.hz
    alpha  = paramCh.alpha  
    D      = paramCh.D      
    gamma  = paramCh.gamma 
    Fc     = paramCh.Fc     
    amp    = paramCh.amp   
    NF     = paramCh.NF

    # channel parameters  
    c_kms = const.c/1e3 # speed of light (vacuum) in km/s
    λ  = c_kms/Fc    # [km]
    α  = alpha/(10*np.log10(np.exp(1))) # [1]
    β2 = -(D*λ**2)/(2*np.pi*c_kms)      # []
    γ  = -gamma

    # generate frequency axis 
    Nfft = len(Ei)
    ω = 2*np.pi*Fs*fftfreq(Nfft)
    
    Nspans = int(np.floor(Ltotal/Lspan))
    Nsteps = int(np.floor(Lspan/hz))
    
    Ech = Ei.reshape(len(Ei),)  

    if order == 1:
        # define linear operator
        linOperator = jnp.exp(-(α/2)*(hz) - 1j*(β2/2)*(ω**2)*(hz))
        @jax.jit
        def one_step(Ech , _):
            # First linear step (frequency domain)
            Ech = fft(Ech)
            Ech = Ech * linOperator            
            Ech = ifft(Ech)
            Ech = Ech * jnp.exp(1j*γ*(Ech*jnp.conj(Ech))*hz)  
            return Ech, None
    elif order == 2:
        # define linear operator
        linOperator = jnp.exp(-(α/2)*(hz/2) - 1j*(β2/2)*(ω**2)*(hz/2))
        @jax.jit
        def one_step(Ech , _):
            # First linear step (frequency domain)
            Ech = Ech * linOperator            

            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech * jnp.exp(1j*γ*(Ech*jnp.conj(Ech))*hz)

            # Second linear step (frequency domain)
            Ech = fft(Ech)       
            Ech = Ech * linOperator   
            return Ech, None
    else:
        raise(ValueError)
    
    myEDFA = partial(edfa, Fs=Fs, G=alpha*Lspan, NF=NF, Fc=Fc)

    if order == 1:
        @jax.jit
        def one_span(carry, _):
            Ech, key = carry
            Ech = jax.lax.scan(one_step, Ech, None,  length=Nsteps)[0]

            if amp =='edfa':
                key, key1 = jax.random.split(key)
                Ech = myEDFA(key1, Ech)
            elif amp =='ideal':
                Ech = Ech * jnp.exp(α/2*Nsteps*hz)
            elif amp == None:
                Ech = Ech * jnp.exp(0)
            return (Ech,key), None
    elif order == 2:
        @jax.jit
        def one_span(carry, _):
            Ech, key = carry
            Ech =  fft(Ech)
            Ech = jax.lax.scan(one_step, Ech, None,  length=Nsteps)[0]
            Ech = ifft(Ech)

            if amp =='edfa':
                key, key1 = jax.random.split(key)
                Ech = myEDFA(key1, Ech)
            elif amp =='ideal':
                Ech = Ech * jnp.exp(α/2*Nsteps*hz)
            elif amp == None:
                Ech = Ech * jnp.exp(0)
            return (Ech,key), None
    else:
        raise(ValueError)

    Ech = jax.lax.scan(one_span, (Ech,key), None, length=Nspans)[0][0]
    
    return Ech.reshape(len(Ech),)


@partial(jax.jit, static_argnums=(2,3))
def manakov_ssf(key, Ei, Fs, paramCh):      
    """
    Manakov model split-step Fourier (symmetric, dual-pol.)

    :param Ei: input signal
    :param Fs: sampling frequency of Ei [Hz]
    :param paramCh: object with physical parameters of the optical channel
    
    :paramCh.Ltotal: total fiber length [km][default: 400 km]
    :paramCh.Lspan: span length [km][default: 80 km]
    :paramCh.hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :paramCh.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :paramCh.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :paramCh.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :paramCh.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    :paramCh.amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    :paramCh.NF: edfa noise figure [dB] [default: 4.5 dB]    
    
    :return Ech: propagated signal
    '''
        E_z = -1/2*alpha*E + j beta2/2 E_tt - j gamma |E|^2E
    '''
    """
    # check input parameters 
    Ltotal = paramCh.Ltotal 
    Lspan  = paramCh.Lspan
    hz     = paramCh.hz
    alpha  = paramCh.alpha  
    D      = paramCh.D      
    gamma  = paramCh.gamma 
    Fc     = paramCh.Fc     
    amp    = paramCh.amp   
    NF     = paramCh.NF

    # channel parameters  
    c_kms = const.c / 1e3 # speed of light (vacuum) in km/s
    λ  = c_kms / Fc
    α  = alpha/(10*np.log10(np.exp(1)))
    β2 = -(D*λ**2)/(2*np.pi*c_kms)
    γ  = -gamma

    # generate frequency axis 
    Nfft = len(Ei)
    ω = 2*jnp.pi*Fs*fftfreq(Nfft)
    
    Nspans = int(np.floor(Ltotal/Lspan))
    Nsteps = int(np.floor(Lspan/hz))
    
    # define linear operator
    linOperator = jnp.exp(-(α/2)*(hz/2) - 1j*(β2/2)*(ω**2)*(hz/2))

    @jax.jit
    def one_step(Ei , _):

        # First linear step (frequency domain) 
        Ei = Ei * linOperator[:,None]      

        # Nonlinear step (time domain)
        # Ei [L,2]
        Ei = ifft(Ei, axis=0)
        Ei = Ei * jnp.exp(1j*(8/9)*γ* jnp.sum(Ei*jnp.conj(Ei), axis=1)[:,None] * hz)

        # Second linear step (frequency domain)
        Ei = fft(Ei, axis=0)       
        Ei = Ei * linOperator[:,None]   
        return Ei, None

    myEDFA = partial(edfa, Fs=Fs, G=alpha*Lspan, NF=NF, Fc=Fc)

    @jax.jit
    def one_span(carry, _):
        Ei, key = carry
        Ei =  fft(Ei, axis=0)
        Ei = jax.lax.scan(one_step, Ei, None,  length=Nsteps)[0]
        Ei = ifft(Ei, axis=0)

        if amp =='edfa':
            key, k1 = jax.random.split(key)
            Ei = myEDFA(k1, Ei)
        elif amp =='ideal':
            Ei = Ei * jnp.exp(α/2*Nsteps*hz)
        elif amp == None:
            Ei = Ei * jnp.exp(0)
        return (Ei, key), None

    Ech = jax.lax.scan(one_span, (Ei, key), None, length=Nspans)[0][0]

    return Ech

def phaseNoise(key, lw, Nsamples, Ts):
    
    σ2 = 2*np.pi*lw*Ts    
    phi = jax.random.normal(key,(Nsamples,),jnp.float32) * jnp.sqrt(σ2)
  
    return jnp.cumsum(phi)


@partial(jax.jit, static_argnums=(2,3,4))
def cssfm(key, Ei, Fs, paramCh, freqSpec=50e9):      
    """
    Split-step Fourier method (symmetric, single-pol.)

    :param Ei: input signal
    :param Fs: sampling frequency of Ei [Hz]
    :param paramCh: object with physical parameters of the optical channel
    “‘
    :paramCh.Ltotal: total fiber length [km][default: 400 km]
    :paramCh.Lspan: span length [km][default: 80 km]
    :paramCh.hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :paramCh.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :paramCh.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :paramCh.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :paramCh.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    :paramCh.amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    :paramCh.NF: edfa noise figure [dB] [default: 4.5 dB]    
    
    :return Ech: propagated signal
    """
    # check input parameters
    Ltotal = paramCh.Ltotal 
    Lspan  = paramCh.Lspan
    hz     = paramCh.hz
    alpha  = paramCh.alpha  
    D      = paramCh.D      
    gamma  = paramCh.gamma 
    Fc     = paramCh.Fc     
    amp    = paramCh.amp   
    NF     = paramCh.NF

    # channel parameters  
    c_kms = const.c/1e3 # speed of light (vacuum) in km/s
    λ  = c_kms/Fc
    α  = alpha/(10*np.log10(np.exp(1)))
    β2 = -(D*λ**2)/(2*np.pi*c_kms)
    γ  = -gamma

    # generate frequency axis 
    Nfft = len(Ei)
    ω = 2*np.pi*Fs*fftfreq(Nfft)
    
    Nspans = int(np.floor(Ltotal/Lspan))
    Nsteps = int(np.floor(Lspan/hz))
    
    Ech = Ei   # L x modes x Nch 

    # define linear operator
    L = Ech.shape[0]
    modes = Ech.shape[1]
    Nch = Ech.shape[2]
    linOperator = np.zeros([L,Nch],np.complex64)
    dω = 2*np.pi*(np.arange(Nch) - (Nch//2)) * freqSpec
    for i in range(Nch):
        linOperator[:,i] = np.exp(-(α/2)*(hz/2) - 1j*(β2*dω[i])*ω*(hz/2) - 1j*(β2/2)*(ω**2)*(hz/2)) # 
    
    linOperator = jnp.repeat(linOperator[:,None,:], modes, axis=1)  # [L, modes, Nch]

    @jax.jit
    def one_step(Ei , _):

        # First linear step (frequency domain) 
        Ei = Ei * linOperator     

        # Nonlinear step (time domain)
        Ei = ifft(Ei, axis=0)

        power = Ei * jnp.conj(Ei)
        P = jnp.sum(power, axis=(1,2))
        if modes == 2:
            Ei = Ei*np.exp((8j/9)*γ*P[:,None,None]*hz)
        else:
            P_rot = 2*P[:,None,None] - power
            Ei = Ei*np.exp((1j)*γ*P_rot*hz)

        # Second linear step (frequency domain)
        Ei = fft(Ei, axis=0)       
        Ei = Ei * linOperator  
        return Ei, None

    myEDFA = partial(edfa, Fs=Fs, G=alpha*Lspan, NF=NF, Fc=Fc)

    @jax.jit
    def one_span(carry, _):
        Ei, key = carry
        Ei =  fft(Ei, axis=0)
        Ei = jax.lax.scan(one_step, Ei, None,  length=Nsteps)[0]
        Ei = ifft(Ei, axis=0)

        if amp =='edfa':
            key, key1 = jax.random.split(key)
            Ei = myEDFA(key1, Ei)
        elif amp =='ideal':
            Ei = Ei * jnp.exp(α/2*Nsteps*hz)
        elif amp == None:
            Ei = Ei * jnp.exp(0)
        return (Ei,key), None

    Ech = jax.lax.scan(one_span, (Ei, key), None, length=Nspans)[0][0]
          
    return Ech

