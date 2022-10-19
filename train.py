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
num = 0
data_sml_tr, paramRx, noise = Rx_data(rd.PRNGKey(0), y_batch[num,:,None], symbWDM[num], 2, param=param, paramCh=paramCh, FO=0, lw=0)

# test data
path_tx = path_ts + f'/Tx_ch19_power{p}'
path_rx = path_ts + f'/Channel_ch19_power{p}_dz0.002'
x_batch, symbWDM, param = pickle.load(open(path_tx, 'rb'))
y_batch, paramCh = pickle.load(open(path_rx, 'rb'))
num = 0
data_sml_ts, paramRx, noise = Rx_data(rd.PRNGKey(0), y_batch[num,:,None], symbWDM[num], 2, param=param, paramCh=paramCh, FO=0, lw=0)

print(f'data.y.shape:{data_sml_tr.y.shape}')

# Define optimizer
# lr1 = optax.piecewise_constant_schedule(1e-4,{2000:1e-5, 4000:1e-6})
# lr2 = optax.warmup_cosine_decay_schedule(
#   init_value=0.0,
#   peak_value=1e-3,
#   warmup_steps=100,
#   decay_steps=4000,
#   end_value=1e-6,
# )
# tx = optax.adam(learning_rate=lr2)

lrD = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3,warmup_steps=100,decay_steps=4000,end_value=1e-6)
lrN = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3,warmup_steps=100,decay_steps=4000,end_value=1e-6)
lrO = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-2,warmup_steps=100,decay_steps=4000,end_value=1e-6)

def fn(k,v):
    if k == 'dispersion_kernel':
        return 'D'
    elif k == 'nonlinear_kernel':
        return 'NL'
    else:
        return 'normal'

from flax.core.frozen_dict import FrozenDict

def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict'''
  def map_fn(nested_dict):
    return FrozenDict({k: (map_fn(v) if isinstance(v, FrozenDict) else fn(k, v))
            for k, v in nested_dict.items()})
  return map_fn

label_fn = map_nested_fn(fn)
tx = optax.multi_transform({'D':optax.adam(learning_rate=lrD), 'NL':optax.adam(learning_rate=lrN), 'normal':optax.adam(learning_rate=lrO)}, label_fn)


## define model
batch_size = 100
sparams_flatkeys = [('DBP',)] # [('DBP',),('RConv',), ('MIMOAF,)]
steps = 1
k = int(1500*(25/steps))
dtaps =  k + (k%2) + 1
ntaps = 1
rtaps = 41
xi = 0
NL = True  # use nonlinear layer or not.
model_train = base.model_init(data_sml_tr, init_len=60000, sparams_flatkeys=sparams_flatkeys, mode='train', steps=steps, xi=xi,  dtaps=dtaps, ntaps=ntaps, rtaps=rtaps, nonlinear_layer=NL)  
model_test = base.model_init(data_sml_ts, init_len=60000, sparams_flatkeys=sparams_flatkeys, mode='test', steps=steps, xi=xi, dtaps=dtaps, ntaps=ntaps, rtaps=rtaps, nonlinear_layer=NL)  

# Training FDBP Model on a single signal 
gen = base.train(model_train, data_sml_tr, batch_size=batch_size, n_iter=6000, tx=tx)
loss0, Train0 = base.run_result(gen)
print('Final train loss:',loss0[-1])
np.save('loading/loss0', loss0)
pickle.dump(Train0[-1].params, open('loading/params','wb'))



# test model
metric1,sig_list = base.test(model_test, Train0[-1].params, data_sml_tr, eval_range=(100,-100))
metric2,sig_list = base.test(model_test, Train0[-1].params, data_sml_ts, eval_range=(100,-100))

# train BER
print('\n training metric \n')
print(metric1)

# test BER
print('\n testing metric \n')
print(metric2)