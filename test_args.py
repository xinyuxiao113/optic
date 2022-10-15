import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt
from optical_flax.dsp import simple_dsp, mimo_dsp
from optical_flax.fiber_system import get_data
import matplotlib.pyplot as plt
import optical_flax.base as base
import optax
from gdbp import gdbp_base as gb, data as gdat, aux
import pickle


path =  '/home/xiaoxinyu/data/0912train_dz_2m'
p = 3.0
path_tx = path + f'/Tx_ch19_power{p}'
path_rx = path + f'/Channel_ch19_power{p}_dz0.002'
x_batch, symbWDM, param = pickle.load(open(path_tx, 'rb'))
y_batch, paramCh = pickle.load(open(path_rx, 'rb'))

from optical_flax.fiber_system import Rx_data
data_sml, paramRx, noise = Rx_data(rd.PRNGKey(0), y_batch[0,:,None], symbWDM[0], 2, param=param, paramCh=paramCh, FO=0, lw=0)

## Training FDBP Model on a signal 
from commplax import optim 
lr = optim.piecewise_constant([500, 1000], [1e-5, 1e-5, 1e-6])
tx = optax.adam(learning_rate=lr)

## define model
batch_size = 100
sparams_flatkeys = []  # [('FDBP',),('RConv',)] static parameters
model_train = base.model_init(data_sml, init_len=40000, sparams_flatkeys=sparams_flatkeys, mode='train', steps=25, xi=1.1, dtaps=1301, ntaps=61, rtaps=61)  
model_test = base.model_init(data_sml, init_len=40000, sparams_flatkeys=sparams_flatkeys, mode='test', steps=25, xi=1.1, dtaps=1301, ntaps=61, rtaps=61)  

# train model
gen = base.train(model_train, data_sml, batch_size=500, n_iter=200, tx=tx)
loss0, Train0 = base.run_result(gen)
