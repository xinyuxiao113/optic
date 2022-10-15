import jax, jax.numpy as jnp, matplotlib.pyplot as plt, jax.random as rd,numpy as np
from optical_flax.dsp import cpr, bps
from optical_flax.fiber_tx import QAM
from optical_flax.fiber_system import Tx_data
from optical_flax.utils import show_symb
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'
from jax.config import config
config.update("jax_enable_x64", True)


## Step 1: data
M = 16
mod = QAM(M)
constSymb = mod.constellation / jnp.sqrt(mod.Es)
sigWDM, symbWDM, param = Tx_data(rd.PRNGKey(0), 1, Nmodes=1, Power=0,Nch=3, SpS=32, Nbits=int(np.log2(M))*4000, Rs=190e9, freq_space=220e9, M=M)
truth = jax.device_get(symbWDM[:,0,:])
n1 = rd.normal(rd.PRNGKey(0), truth.shape, dtype=jnp.float64)*0.05
n2 = rd.normal(rd.PRNGKey(1), truth.shape, dtype=jnp.complex128)*0.1
pn = jnp.cumsum(n1, axis=0)
y = truth * jnp.exp(1j*pn) + n2

## Step 2: Adaptive filter
## BPS
Eo, theta = bps(jax.device_get(y),5, jax.device_get(constSymb), 101)
