import jax.numpy as jnp
from commplax.xop import convolve, frame


from jax.numpy.fft import fft,fftfreq,fftshift

def P(x):
    '''
    镜像对称
    '''
    return jnp.roll(jnp.flip(x),1)

def ifft(x, axis=-1):
    x = jnp.fft.fft(x, axis=axis)
    N = x.shape[axis]
    return 1/N*P(x)


# convolve = jnp.convolve

def firFilter(h, x):
    """
    Implements FIR filtering and compensates filter delay
    (assuming the impulse response is symmetric)
    零边值的中心卷积
    
    :param h: impulse response (symmetric)
    :param x: input signal 
    :return y: output filtered signal    
    """   
    N = h.size
    x = jnp.pad(x, (0, int(N/2)),'constant')
    #y = lfilter(h,1,x)
    y = convolve(h,x)[0:x.size]
    
    return y[int(N/2):y.size]


def circFilter(h, x):
    ''' 
        周期边界的卷积   h的中心为0
    '''
    k = h.shape[0]//2
    N = x.shape[0]
    y = convolve(x,h)
    z = y[k:k + N ]
    s = y[k+N:]
    z = z.at[0:s.size].add(s)
    z = z.at[-k:].add(y[0:k])
    return z


from optical_flax.utils import conv_circ
def circFilter_(h,x):
    k = h.size // 2
    h_ = jnp.pad(h,(0,x.size - h.size), 'constant')
    h_ = jnp.roll(h_, -k)
    return conv_circ(h_,x)