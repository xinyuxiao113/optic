
import jax
import numpy as np
from jax import numpy as jnp
from commplax import comm, xcomm, xop, adaptive_filter as af
from commplax.module import core
from typing import Any, Optional, Union
import jax.random as random


def zeros(key, shape, dtype=jnp.float32): 
    return jnp.zeros(shape, dtype)


def ones(key, shape, dtype=jnp.float32): return jnp.ones(shape, dtype)


def delta(key, shape, dtype=jnp.float32):
    k1d = comm.delta(shape[0], dtype=dtype)
    return jnp.tile(np.expand_dims(k1d, axis=list(range(1, len(shape)))), (1,) + shape[1:])




def gauss(key, shape, dtype=jnp.float32):
    taps = shape[0]
    k1d = comm.gauss(comm.gauss_minbw(taps), taps=taps, dtype=dtype)
    return jnp.tile(np.expand_dims(k1d, axis=list(range(1, len(shape)))), (1,) + shape[1:])


def near_zeros(key, shape, dtype=jnp.float32):
    return random.normal(key, shape, dtype) * 1e-2


def fdbp_init(a: dict,
              xi: float = 1.1,
              steps: Optional[int] = None,
              domain: str = 'time'):
    '''
        initializer for the base module

        Args:
            xi: NLC scaling factor
            steps: GDBP steps, used to calculate the theoretical profiles of D- and N-filters
            domain: 'time' or 'frequency'

        Returns:
            a pair of functions to initialize D- and N-filters
    '''

    def d_init(key, shape, dtype=jnp.complex64):
        dtaps = shape[0]
        d0, _ = dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,  # rescale as input power which has been norm to 2 in dataloader
            virtual_spans=steps,
            carrier_frequency=a['carrier_frequency'],
            fiber_dispersion=a['D'],
            fiber_loss=a['fiber_loss'],
            gamma=a['gamma'],
            domain=domain,
            ignore_beta3=True)
        return d0[0, :, 0]

    def n_init(key, shape, dtype=jnp.float32):
        dtaps = shape[0]
        _, n0 = dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,  # rescale as input power which has been norm to 2 in dataloader
            virtual_spans=steps,
            carrier_frequency=a['carrier_frequency'],
            fiber_dispersion=a['D'],
            fiber_loss=a['fiber_loss'],
            gamma=a['gamma'],
            domain=domain,
            ignore_beta3=True)

        return xi * n0[0, 0, 0] * core.gauss(key, shape, dtype)

    return d_init, n_init


def dbp_params(
    sample_rate,                                      # sample rate of target signal [Hz]
    span_length,                                      # length of each fiber span [m]
    spans,                                            # number of fiber spans
    freqs,                                            # resulting size of linear operator
    launch_power=0,                                   # launch power [dBm]
    steps_per_span=1,                                 # steps per span
    virtual_spans=None,                               # number of virtual spans
    carrier_frequency=299792458/1550E-9,              # carrier frequency [Hz]
    fiber_dispersion=16.5E-6,                         # [s/m^2]
    fiber_dispersion_slope=0.08e3,                    # [s/m^3]
    fiber_loss=.2E-3,                                 # loss of fiber [dB/m]
    gamma = 1.6567e-3,                                # 1/W/m             
    fiber_reference_frequency=299792458/1550E-9,      # fiber reference frequency [Hz]
    ignore_beta3=False,
    polmux=True,
    domain='time',
    step_method="uniform"):

    domain = domain.lower()
    assert domain == 'time' or domain == 'frequency'

    # short names
    pi  = np.pi
    log = np.log
    exp = np.exp
    ifft = np.fft.ifft
    

    # virtual span is used in cases where we do not use physical span settings
    if virtual_spans is None:
        virtual_spans = spans

    C       = 299792458. # speed of light [m/s]
    lambda_ = C / fiber_reference_frequency
    B_2     = -fiber_dispersion * lambda_**2 / (2 * pi * C)
    B_3     = 0. if ignore_beta3 else (fiber_dispersion_slope * lambda_**2 + 2 * fiber_dispersion * lambda_) * (lambda_ / (2 * pi * C))**2 #[/m/W]
    LP      = 10.**(launch_power / 10 - 3)  # 将[dBm]转化为 [W]
    alpha   = fiber_loss / (10. / log(10.)) # 计算出方程中的衰减系数alpha，z单位 [dB/m]
    L_eff   = lambda h: (1 - exp(-alpha * h)) / alpha
    NIter   = virtual_spans * steps_per_span
    delay   = (freqs - 1) // 2
    dw      = 2 * pi * (carrier_frequency - fiber_reference_frequency)
    w_res   = 2 * pi * sample_rate / freqs
    k       = np.arange(freqs)
    w       = np.where(k > delay, k - freqs, k) * w_res # ifftshifted

    if step_method.lower() == "uniform":
        dz = span_length * spans / virtual_spans / steps_per_span
        H   = exp(-1j * (-B_2 / 2 * (w + dw)**2 + B_3 / 6 * (w + dw)**3) * dz)
        H_casual = H * exp(-1j * w * delay / sample_rate) ## 频域相位旋转等价于时域平移，将时域对齐
        h_casual = ifft(H_casual)
        ## 下面是正号
        phi = spans / virtual_spans * gamma * L_eff(span_length / steps_per_span) * LP * \
            exp(-alpha * span_length * (steps_per_span - np.arange(0, NIter) % steps_per_span-1) / steps_per_span)
    else:
        raise ValueError("step method '%s' not implemented" % step_method)

    if polmux:
        dims = 2
    else:
        dims = 1

    H = np.tile(H[None, :, None], (NIter, 1, dims))
    h_casual = np.tile(h_casual[None, :, None], (NIter, 1, dims))
    phi = np.tile(phi[:, None, None], (1, dims, dims))

    return (h_casual, phi) if domain == 'time' else (H, phi)
