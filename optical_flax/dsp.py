import jax, numpy as np, jax.numpy as jnp
from scipy.stats.kde import gaussian_kde
import scipy.constants as const
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import jax
from scipy import signal
from jax import device_get

from commpy.filters import rrcosfilter, rcosfilter
from commpy.utilities  import signal_power, upsample
from commpy.modulation import QAMModem

from commplax.xop import convolve
from commplax import equalizer as eq 
from commplax import comm
from commplax.module import core

from optical_flax.fiber_rx import linFiberCh
from optical_flax.core import parameters
from optical_flax.operator import fft, ifft, fftfreq, fftshift, auto_rho



def eyediagram(sig, Nsamples, SpS, n=3, ptype='fast', plotlabel=None):
    """
    Plots the eye diagram of a modulated signal waveform

    :param Nsamples: number os samples to be plotted
    :param SpS: samples per symbol
    :param n: number of symbol periods
    :param type: 'fast' or 'fancy'
    :param plotlabel: label for the plot legend
    """
    
    if np.iscomplex(sig).any():
        d = 1
        plotlabel_ = plotlabel+' [real]'
    else:
        d = 0
        plotlabel_ = plotlabel
        
    for ind in range(0, d+1):
        if ind == 0:
            y = sig[0:Nsamples].real
            x = np.arange(0,y.size,1) % (n*SpS)            
        else:
            y = sig[0:Nsamples].imag
            plotlabel_ = plotlabel+' [imag]'       
     
        plt.figure();
        if ptype == 'fancy':            
            k = gaussian_kde(np.vstack([x, y]))
            k.set_bandwidth(bw_method=k.factor/5)

            xi, yi = 1.1*np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, shading='auto');
            plt.show();
        elif ptype == 'fast':
            y[x == n*SpS] = np.nan;
            y[x == 0] = np.nan;
            
            plt.plot(x/SpS, y, color='blue', alpha=0.8, label=plotlabel_);
            plt.xlim(min(x/SpS), max(x/SpS))
            plt.xlabel('symbol period (Ts)')
            plt.ylabel('amplitude')
            plt.title('eye diagram')
            
            if plotlabel != None:
                plt.legend(loc='upper left')
                
            plt.grid()
            plt.show();
    return None
    
def sincInterp(x, fa):
    
    fa_sinc = 32*fa
    Ta_sinc = 1/fa_sinc
    Ta = 1/fa
    t = np.arange(0, x.size*32)*Ta_sinc
    
    plt.figure()  
    y = upsample(x,32)
    y[y==0] = np.nan
    plt.plot(t,y.real,'ko', label='x[k]')
    
    x_sum = 0
    for k in range(0, x.size):
        xk_interp = x[k]*np.sinc((t-k*Ta)/Ta)
        x_sum += xk_interp
        plt.plot(t, xk_interp)           
    
    plt.legend(loc="upper right")
    plt.xlim(min(t), max(t))
    plt.grid()
    
    return x_sum, t

def lowPassFIR(fc, fa, N, typeF = 'rect'):
    """
    Calculate FIR coeffs for a lowpass filter
    
    :param fc : cutoff frequency
    :param fa : sampling frequency
    :param N  : number of coefficients
    :param typeF : 'rect' or 'gauss'
    
    :return h : FIR filter coefficients
    """
    fu = fc/fa    
    d  = (N-1)/2    
    n  = np.arange(0, N)    
  
    # calculate filter coefficients
    if typeF == 'rect':
        h = (2*fu)*np.sinc(2*fu*(n-d))
    elif typeF == 'gauss':    
        h = np.sqrt(2*np.pi/np.log(2))*fu*np.exp(-(2/np.log(2))*(np.pi*fu*(n-d))**2)
    
    return h

def edc(Ei, L, D, Fc, Fs):
    """
    Electronic chromatic dispersion compensation (EDC)

    :param Ei: dispersed signal
    :param L: fiber length [km]    
    :param D: chromatic dispersion parameter [ps/nm/km]   
    :param Fc: carrier frequency [Hz]
    :param Fs: sampling frequency [Hz]
    
    :return Eo: CD compensated signal
    """
    Eo, H = linFiberCh(Ei, L, 0, -D, Fc, Fs)
    
    return Eo, H

def cpr(Ei, N, constSymb, symbTx):    
    """
    Carrier phase recovery (CPR) for single mode.
    
    """    
    nModes = Ei.shape[1]
    ϕ  = np.zeros(Ei.shape)    
    θ  = np.zeros(Ei.shape)
    
    for i in range(nModes):
        for k in range(0,len(Ei)):
            
            predict = Ei[k,i]*np.exp(1j*θ[k-1,i])
            decided = np.argmin(np.abs(predict - constSymb)) # find closest constellation symbol
            
            if k % 50 == 0:
                ϕ[k,i] = np.angle(symbTx[k,i]/predict) + θ[k-1,i] # phase estimation with pilot symbol
            else:
                ϕ[k,i] = np.angle(constSymb[decided]/predict) + θ[k-1,i] # phase estimation after symbol decision
                    
            if k > N:
                θ[k,i]  = np.mean(ϕ[k-N:k+1,i]) # moving average filter
            else:           
                θ[k,i] = np.angle(symbTx[k,i]/predict) + θ[k-1,i]
                
    
    Eo = Ei*np.exp(1j*θ) # compensate phase rotation
        
    return Eo, θ



def fourthPowerFOE(Ei, Ts, plotSpec=False):
    """
    4th power frequency offset estimator (FOE)    
    """
        
    Fs = 1/Ts
    Nfft = len(Ei)
    
    f = Fs*fftfreq(Nfft)
    f = fftshift(f)
    
    f4 = 10*np.log10(np.abs(fftshift(fft(Ei**4))))    
    indFO = np.argmax(f4)

    if plotSpec:
        plt.figure()
        plt.plot(f, f4, label = '$|FFT(s[k]^4)|[dB]$')
        plt.plot(f[indFO], f4[indFO],'x',label='$4f_o$')
        plt.legend()
        plt.xlim(min(f), max(f))
        plt.grid()
    
    return f[indFO]/4

def dbp(Ei, Fs, Ltotal, Lspan, hz=0.5, alpha=0.2, gamma=1.3, D=16, Fc=193.1e12):      
    """
    Digital backpropagation (symmetric, single-pol.)

    :param Ei: input signal
    :param Ltotal: total fiber length [km]
    :param Lspan: span length [km]
    :param hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :param alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :param gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :param Fc: carrier frequency [Hz][default: 193.1e12 Hz]
    :param Fs: sampling frequency [Hz]

    :return Ech: backpropagated signal
    """           
    #c = 299792458   # speed of light (vacuum)
    c_kms = const.c/1e3
    λ  = c_kms/Fc
    α  = -alpha/(10*np.log10(np.exp(1)))
    β2 = (D*λ**2)/(2*np.pi*c_kms)
    γ  = -gamma
            
    Nfft = len(Ei)

    ω = 2*np.pi*Fs*fftfreq(Nfft)
    
    Nspans = int(np.floor(Ltotal/Lspan))
    Nsteps = int(np.floor(Lspan/hz))   
        
    Ech = Ei.reshape(len(Ei),)    
    Ech = fft(Ech) #single-polarization field    
    
    linOperator = np.exp(-(α/2)*(hz/2) - 1j*(β2/2)*(ω**2)*(hz/2))
        
    for spanN in tqdm(range(0, Nspans)):
        
        Ech = Ech*np.exp((α/2)*Nsteps*hz)
                
        for stepN in range(0, Nsteps):            
            # First linear step (frequency domain)
            Ech = Ech*linOperator            
                      
            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech*np.exp(1j*γ*(Ech*np.conj(Ech))*hz)
            
            # Second linear step (frequency domain)
            Ech = fft(Ech)       
            Ech = Ech*linOperator             
                
    Ech = ifft(Ech) 
       
    return Ech.reshape(len(Ech),)


def downsampling(sigRx, down_rate, discard=100):
    '''
    sigRx: [N,2]
    '''
    # finds best sampling instant
    varVector = np.var((sigRx.T).reshape(-1,down_rate), axis=0)
    sampDelay = np.where(varVector == np.amax(varVector))[0][0]
    sigRx = sigRx[sampDelay::down_rate]

    ind = np.arange(discard, sigRx.shape[0] - discard)
    sigRx = sigRx/np.sqrt(signal_power(sigRx[ind]))

    return sigRx


def cpr2(Ei, symbTx=[], paramCPR=[]):
    """
    Carrier phase recovery function (CPR)

    Parameters
    ----------
    Ei : complex-valued ndarray
        received constellation symbols.
    symbTx :complex-valued ndarray, optional
        Transmitted symbol sequence. The default is [].
    paramCPR : core.param object, optional
        configuration parameters. The default is [].
        
        BPS params:
            
        paramCPR.alg: CPR algorithm to be used ['bps' or 'ddpll']
        paramCPR.M: constellation order. The default is 4.
        paramCPR.N: length of BPS the moving average window. The default is 35.    
        paramCPR.B: number of BPS test phases. The default is 64.
        
        DDPLL params:
            
        paramCPR.tau1: DDPLL loop filter param. 1. The default is 1/2*pi*10e6.
        paramCPR.tau2: DDPLL loop filter param. 2. The default is 1/2*pi*10e6.
        paramCPR.Kv: DDPLL loop filter gain. The default is 0.1.
        paramCPR.Ts: symbol period. The default is 1/32e9.
        paramCPR.pilotInd: indexes of pilot-symbol locations.

    Raises
    ------
    ValueError
        Error is generated if the CPR algorithm is not correctly
        passed.

    Returns
    -------
    Eo : complex-valued ndarray
        Phase-compensated signal.
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    """

    # check input parameters
    alg = getattr(paramCPR, "alg", "bps")
    M = getattr(paramCPR, "M", 16)
    B = getattr(paramCPR, "B", 64)
    N = getattr(paramCPR, "N", 35)
    Kv = getattr(paramCPR, "Kv", 0.1)
    tau1 = getattr(paramCPR, "tau1", 1 / (2 * np.pi * 10e6))
    tau2 = getattr(paramCPR, "tau2", 1 / (2 * np.pi * 10e6))
    Ts = getattr(paramCPR, "Ts", 1 / 32e9)
    pilotInd = getattr(paramCPR, "pilotInd", np.array([len(Ei) + 1]))

    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)
    mod = QAMModem(m=M)
    constSymb = mod.constellation / np.sqrt(mod.Es)

    if alg == "ddpll":
        θ = ddpll(Ei, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd)
    elif alg == "bps":
        θ = bps(Ei, int(N / 2), constSymb, B)
    else:
        raise ValueError("CPR algorithm incorrectly specified.")
    θ = np.unwrap(4 * θ, axis=0) / 4

    Eo = Ei * np.exp(1j * θ)

    if Eo.shape[1] == 1:
        Eo = Eo[:]
        θ = θ[:]
    return Eo, θ


@njit
def bps(Ei, N, constSymb, B):
    """
    Blind phase search (BPS) algorithm

    Parameters
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols.
    N : int
        Half of the 2*N+1 average window.
    constSymb : complex-valued ndarray
        Complex-valued constellation.
    B : int
        number of test phases.

    Returns
    -------
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    """

    nModes = Ei.shape[1]

    ϕ_test = np.arange(0, B) * (np.pi / 2) / B - np.pi/4  # test phases

    θ = np.zeros(Ei.shape, dtype="float")

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    L = x.shape[0]

    for n in range(0, nModes):

        dist = np.zeros((B, constSymb.shape[0]), dtype="float")
        dmin = np.zeros((B, 2 * N + 1), dtype="float")

        for k in range(0, L):
            for indPhase, ϕ in enumerate(ϕ_test):
                dist[indPhase, :] = np.abs(x[k, n] * np.exp(1j * ϕ) - constSymb) ** 2
                dmin[indPhase, -1] = np.min(dist[indPhase, :])
            if k >= 2 * N:
                sumDmin = np.sum(dmin, axis=1)
                indRot = np.argmin(sumDmin)
                θ[k - 2 * N, n] = ϕ_test[indRot]
            dmin = np.roll(dmin, -1)
    θ = np.unwrap(θ, axis=0, period=np.pi/2) 
    return Ei*jnp.exp(1j*θ), θ

@njit
def ddpll(Ei, Kv, constSymb, symbTx, Ts=1/36e9, tau1=1/1e6, tau2=1/1e6, pilotInd=np.arange(200,dtype=int)):
    """
    Decision-directed Phase-locked Loop (DDPLL) algorithm

    Parameters
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols.
    Ts : float scalar
        Symbol period.
    Kv : float scalar
        Loop filter gain.
    tau1 : float scalar
        Loop filter parameter 1.
    tau2 : float scalar
        Loop filter parameter 2.
    constSymb : complex-valued ndarray
        Complex-valued ideal constellation symbols.
    symbTx : complex-valued ndarray
        Transmitted symbol sequence.
    pilotInd : int ndarray
        Indexes of pilot-symbol locations.

    Returns
    -------
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    References
    -------
    [1] H. Meyer, Digital Communication Receivers: Synchronization, Channel 
    estimation, and Signal Processing, Wiley 1998. Section 5.8 and 5.9.    
    
    """
    nModes = Ei.shape[1]

    θ = np.zeros(Ei.shape)
    if np.ndim(symbTx) == 1:
        symbTx = symbTx[:,None]


    # Loop filter coefficients
    a1b = np.array(
        [
            1,
            Ts / (2 * tau1) * (1 - 1 / np.tan(Ts / (2 * tau2))),
            Ts / (2 * tau1) * (1 + 1 / np.tan(Ts / (2 * tau2))),
        ]
    )

    u = np.zeros(3)  # [u_f, u_d1, u_d]

    for n in range(0, nModes):

        u[2] = 0  # Output of phase detector (residual phase error)
        u[0] = 0  # Output of loop filter

        for k in range(0, len(Ei)-1):
            u[1] = u[2]

            # Remove estimate of phase error from input symbol
            Eo = Ei[k, n] * np.exp(1j * θ[k, n])

            # Slicer (perform hard decision on symbol)
            if k in pilotInd:
                # phase estimation with pilot symbol
                # Generate phase error signal (also called x_n (Meyer))
                u[2] = np.imag(Eo * np.conj(symbTx[k, n])) 
                # u[2] = np.imag(Eo * np.conj(symbTx[k, n])) * np.real(Eo * np.conj(symbTx[k, n]))
            else:
                # find closest constellation symbol
                decided = np.argmin(np.abs(Eo - constSymb))
                # Generate phase error signal (also called x_n (Meyer))
                u[2] = np.imag(Eo * np.conj(constSymb[decided])) 
                # u[2] = np.imag(Eo * np.conj(constSymb[decided])) * np.real(Eo * np.conj(constSymb[decided]))
            # Pass phase error signal in Loop Filter (also called e_n (Meyer))
            u[0] = np.sum(a1b * u)

            # Estimate the phase error for the next symbol
            θ[k + 1, n] = θ[k, n] - Kv * u[0]
    
    θ = np.unwrap(θ, axis=0, period=np.pi/2) 
    return Ei*jnp.exp(1j*θ), θ


def simple_cpr(sigRx, symbTx, discard=100):
    '''
    sigRx: [N,2] have done!
    '''
    ind = np.arange(discard, sigRx.shape[0] - discard)
    rot = np.mean(symbTx[ind]/sigRx[ind], axis=0)
    sigRx  = rot[None,:] * sigRx

    y = []
    for i in range(sigRx.shape[1]):
        y.append(sigRx[:,i]/np.sqrt(signal_power(sigRx[ind,i])))
    return jnp.stack(y, axis=-1)


def test_result(sigRx7, symbTx, mod, discard=100, show_name='Your method'):
    sigRx = sigRx7
    SNR = signal_power(symbTx)/signal_power(sigRx-symbTx)

    # hard decision demodulation of the received symbols    
    bitsRx = mod.demodulate(np.sqrt(mod.Es)*sigRx, demod_type = 'hard') 
    bitsTx = mod.demodulate(np.sqrt(mod.Es)*symbTx, demod_type = 'hard') 

    err = np.logical_xor(bitsRx[discard:bitsRx.size-discard], 
                        bitsTx[discard:bitsTx.size-discard])
    BER = np.mean(err)

    print(f'############################ {show_name} ############################ ')
    print('Estimated SNR = %.2f dB \n'%(10*np.log10(SNR)))
    print('Total counted bits = %d  '%(err.size))
    print('Total of counted errors = %d  '%(err.sum()))
    print('BER = %.2e  '%(BER))
    plt.figure()
    plt.plot(err,'o', label = 'errors location')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(5,5))
    plt.ylabel('$S_Q$', fontsize=14)
    plt.xlabel('$S_I$', fontsize=14)
    plt.grid()
    
    plt.plot(sigRx.real,sigRx.imag,'.', markersize=4, label='Rx')
    plt.plot(symbTx.real,symbTx.imag,'k.', markersize=4, label='Tx')
    return SNR, BER, err


def simple_dsp(data, eval_range=(30000, -20000), metric_fn=comm.qamqot):
    '''
    ## 结果不好的原因可能是重抽样的方式不恰当 ！！！ 应当如何进行重抽样 ？？
    '''
    a = data.a

    # step 1: cdc
    x = data.x
    y0 = data.y
    y1,H = edc(y0, a['distance']/1e3, a['D'], a['Fc'] - 64e6, a['samplerate'])

    # step 2: down sampling + discard, normalization + time recoverry   #############
    discard = 100
    y2 = downsampling(y1, a['sps'] , discard=discard)
    y2, symbDelay = time_recovery_vmap(y2, x)
    y2 = device_get(y2)
    print(f'symb Delay: {symbDelay}')

    # step 3： FOE  如何结合两个方向的FO   ####### 了解带极化方向的相干接收机
    fo = fourthPowerFOE(y2[:,0], 1/a['baudrate']) # 只是用 x 方向估计FO
    y3 = y2 * np.exp(-1j*2 * np.pi * fo * np.arange(0,len(y2))/a['baudrate'])[:,None] 
    print('Estimated FO : %3.4f MHz'%(fo/1e6))

    # step 4: CPR
    paramCPR = parameters()
    paramCPR.alg = 'ddpll'
    paramCPR.M   = a['M']
    paramCPR.tau1 = 1/(2*np.pi*10e3)
    paramCPR.tau2 = 1/(2*np.pi*10e3)
    paramCPR.Kv  = 0.01
    paramCPR.pilotInd = np.arange(0, len(y3), 50)
    y4, theta = cpr2(y3, x, paramCPR)


    # step 5: rotation   
    y5 = simple_cpr(y4, x)

    y0 =  core.Signal(y0, core.SigTime(0,0,2))
    y1 =  core.Signal(y1, core.SigTime(0,0,2))
    y2 =  core.Signal(y2, core.SigTime(0,0,1))
    y3 =  core.Signal(y3, core.SigTime(0,0,1))
    y4 =  core.Signal(y4, core.SigTime(0,0,1))
    y5 =  core.Signal(y5, core.SigTime(0,0,1))
    name = ['Rx', 'cdc','dwn sample','FOE', 'CPR', 'rotation']

    z = y5
    metric = metric_fn(z.val,
                        data.x[z.t.start: data.x.shape[0] + z.t.stop],
                        scale=jnp.sqrt(10),
                        eval_range=eval_range)

    return (y0,y1,y2,y3,y4,y5), name, metric, theta


def cor(x,y):
    res = []
    for i in range(x.shape[1]):
        res.append(signal.correlate(x[:,i],y[:,i]))
    return np.stack(res, axis=-1)


def mimo_dsp(data, eval_range=(30000, -20000), metric_fn=comm.qamqot):
    y = []
    x = data.x
    y.append(data.y)
    y.append(eq.cdcomp(y[0], data.a['samplerate'], CD=data.a['CD']))
    y.append(eq.modulusmimo(y[1], taps=21, lr=2**-14)[0])  # 这一步可能把符合序号映射错
    y.append(time_recovery_vmap(y[2], x)[0])
    y.append(eq.qamfoe(y[3])[0])
    y.append(eq.ekfcpr(y[4])[0])
    y.append(simple_cpr(y[5], x) )

    sig_list = []
    sig_list.append(core.Signal(y[0],core.SigTime(0,0,2)))
    sig_list.append(core.Signal(y[1],core.SigTime(0,0,2)))
    sig_list.append(core.Signal(y[2],core.SigTime(0,0,1)))
    sig_list.append(core.Signal(y[3],core.SigTime(0,0,1)))
    sig_list.append(core.Signal(y[4],core.SigTime(0,0,1)))
    sig_list.append(core.Signal(y[5],core.SigTime(0,0,1)))
    sig_list.append(core.Signal(y[6],core.SigTime(0,0,1)))

    name = ['Rx', 'cdc','mimo','time recovery', 'FOE', 'CPR', 'rotation']

    z = sig_list[-1]
    metric = metric_fn(z.val,
                        data.x[z.t.start:data.x.shape[0] + z.t.stop],
                        scale=jnp.sqrt(10),
                        eval_range=eval_range)

    return sig_list, name, metric


def time_recovery(y,x):
    '''
    predict sequence: y
    truth: x
    '''
    i = jnp.argmax(auto_rho(jnp.abs(x), jnp.abs(y)).real)
    return jnp.roll(y, i), i

time_recovery_vmap = jax.vmap(time_recovery, in_axes=-1, out_axes=-1)


