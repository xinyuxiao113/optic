import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.50'
import warnings
warnings.filterwarnings("ignore")

import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt
from optical_flax.dsp import simple_dsp, mimo_dsp
from optical_flax.fiber_system import get_data
import matplotlib.pyplot as plt
import optical_flax.base as base
import optax
from gdbp import gdbp_base as gb, data as gdat, aux
import pickle
from optical_flax.fiber_system import Rx_data



path_tr =  '/home/xiaoxinyu/data/0912train_dz_2m'
path_ts = '/home/xiaoxinyu/data/0912test_dz_2m'
p = 3.0

# train data
path_tx = path_tr + f'/Tx_ch19_power{p}'
path_rx = path_tr + f'/Channel_ch19_power{p}_dz0.002'
x_batch, symbWDM, param = pickle.load(open(path_tx, 'rb'))
y_batch, paramCh = pickle.load(open(path_rx, 'rb'))
data_sml_tr, paramRx, noise = Rx_data(rd.PRNGKey(0), y_batch[0,:,None], symbWDM[0], 2, param=param, paramCh=paramCh, FO=0, lw=0)

# test data
path_tx = path_ts + f'/Tx_ch19_power{p}'
path_rx = path_ts + f'/Channel_ch19_power{p}_dz0.002'
x_batch, symbWDM, param = pickle.load(open(path_tx, 'rb'))
y_batch, paramCh = pickle.load(open(path_rx, 'rb'))
num = 0
data_sml_ts, paramRx, noise = Rx_data(rd.PRNGKey(0), y_batch[num,:,None], symbWDM[num], 2, param=param, paramCh=paramCh, FO=0, lw=0)


# Define optimizer
lr1 = optax.piecewise_constant_schedule(1e-4,{2000:1e-5, 4000:1e-6})
lr2 = optax.warmup_cosine_decay_schedule(
  init_value=0.0,
  peak_value=1e-3,
  warmup_steps=100,
  decay_steps=4000,
  end_value=1e-6,
)
tx = optax.adam(learning_rate=lr2)

# tx = optax.adamw(learning_rate=lr2, weight_decay=0.01)
# tx = optax.chain(
#   optax.clip(1.0),
#   optax.adamw(learning_rate=schedule,weight_decay=0.0001),
# )

## define model
batch_size = 100

sparams_flatkeys = [] # [('DBP',),('RConv',)]   # [('DBP',),('RConv',)] static parameters
steps = 1
k = int(1500*(25/steps))
dtaps =  k + (k%2) + 1
ntaps = 61
rtaps = 41
xi = 0
NL = True
model_train = base.model_init(data_sml_tr, init_len=60000, sparams_flatkeys=sparams_flatkeys, mode='train', steps=steps, xi=xi,  dtaps=dtaps, ntaps=ntaps, rtaps=rtaps, nonlinear_layer=NL)  
model_test = base.model_init(data_sml_ts, init_len=60000, sparams_flatkeys=sparams_flatkeys, mode='test', steps=steps, xi=xi, dtaps=dtaps, ntaps=ntaps, rtaps=rtaps, nonlinear_layer=NL)  

# Training FDBP Model on a single signal 
gen = base.train(model_train, data_sml_tr, batch_size=500, n_iter=6000, tx=tx)
loss0, Train0 = base.run_result(gen)
print('Final train loss:',loss0[-1])
np.save('loading/loss0', loss0)
pickle.dump(Train0[-1].params, open('loading/params','wb'))


# test model
metric,sig_list = base.test(model_test, Train0[-1].params, data_sml_ts, eval_range=(100,-100))
print(metric)
print()