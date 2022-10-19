import jax.numpy as jnp
import jax
import numpy as np
from jax.numpy.fft import fft,fftfreq,fftshift

from commplax.xop import convolve


def L2(x):
    '''
    [sqrt(W)]
    '''
    return jnp.sqrt(jnp.mean(jnp.abs(x)**2))


def P(x):
    '''
    Mirror operator. Use for ifft
    Input:
        x=[x0,x1,...,x_{N-1}]
    Output:
        Px=[x0,x_{N-1},x_{N-2},...,x_2]
    '''
    return jnp.roll(jnp.flip(x),1)


# 使用 jnp.fft.ifft 会造成误差  #BUG: fft(ifft) != id, 在float64下可以看出区别
def ifft(x, axis=-1):
    '''
        Same as jnp.fft.ifft
    Input:
        x: array.
        axis: ifft axis.
    Output:
        ifft(x): ifft Array of x on the specified axis.
    '''
    x = jnp.fft.fft(x, axis=axis)
    N = x.shape[axis]
    return 1/N*P(x)



def firFilter(h, x):
    """
    1D zeros pad convolutoin.
    
    Input:
        h: 1D convolution kernel.
        x: 1D signal.
    Output:
        y: output filtered signal.
    """   
    N = h.size
    x = jnp.pad(x, (0, int(N/2)),'constant')
    #y = lfilter(h,1,x)
    y = convolve(h,x)[0:x.size]
    
    return y[int(N/2):y.size]


def circFilter(h, x):
    ''' 
        1D Circular convolution.
    '''
    k = h.shape[0]//2
    N = x.shape[0]
    y = convolve(x,h)
    z = y[k:k + N ]
    s = y[k+N:]
    z = z.at[0:s.size].add(s)
    z = z.at[-k:].add(y[0:k])
    return z

def validFilter(h, x):
    '''
        1D valid pad convolution.
    '''
    k = h.shape[0] // 2
    y = circFilter(h, x)
    return y[k:-k]



def circFilter_(h,x):
    k = h.size // 2
    h_ = jnp.pad(h,(0,x.size - h.size), 'constant')
    h_ = jnp.roll(h_, -k)
    return conv_circ(h_,x)


def frame_gen(x, flen, fstep, fnum=None):
    '''
        generate circular frame from Array x.
    '''
    s = np.arange(flen)
    N = x.shape[0]

    if fnum == None:
        fnum = 1 + (N - flen) // fstep
    

    for i in range(fnum):
        yield x[(s + i* fstep) % N]
    

def frame(x, flen, fstep, fnum=None):
    N = x.shape[0]

    if fnum == None:
        fnum = 1 + (N - flen) // fstep
    
    ind = (np.arange(flen)[None,:] + fstep * np.arange(fnum)[:,None]) % N
    return x[ind,...]



def conv_circ( signal, ker ):
    '''
    N-size circular convolution.

    Input:
        signal: real 1D array with shape (N,)
        ker: real 1D array with shape (N,).
    Output:
        signal conv_N  ker.
    '''
    return jnp.fft.ifft( jnp.fft.fft(signal)*jnp.fft.fft(ker) )

def corr_circ(x, y):  # 不交换
    '''
    N-size correlation.
    Input:
        x: 1D array with shape (N,)
        y: 1D array with shape (N,)
    Output:
        z: 1D array with shape (N,)
        z[n] = \sum_{i=1}_{N} x[i] y[i - n] = \sum_{i=1}_{N} x[i + n]y[i]
    '''
    return conv_circ(x, jnp.roll(jnp.flip(y),1))


def auto_rho(x,y):
    '''
        auto-correlation coeff.
    '''
    N = len(x)
    Ex = jnp.mean(x)
    Ey = jnp.mean(y)
    Vx = jnp.var(x)
    Vy = jnp.var(y)
    return (corr_circ(x,y)/N - Ex*Ey)/jnp.sqrt(Vx*Vy)
    

    
