import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.50'
import warnings
warnings.filterwarnings("ignore")
import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt
import optax
import pickle
from optical_flax.fiber_system import Rx_data
from optical_flax import base
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--steps',   help='choose the model steps', type=int)
parser.add_argument('--power',   help='choose the power: -3.0, 0.0, 3.0, 6.0', type=float)
args = parser.parse_args()

path_tr =  '/home/xiaoxinyu/data/0912train_dz_2m'
path_ts = '/home/xiaoxinyu/data/0912test_dz_2m'
p = args.power

# train data
path_tx = path_ts + f'/Tx_ch19_power{p}'
path_rx = path_ts + f'/Channel_ch19_power{p}_dz0.002'
x_batch, symbWDM, param = pickle.load(open(path_tx, 'rb'))
y_batch, paramCh = pickle.load(open(path_rx, 'rb'))
num = 0
data_sml_tr, paramRx, noise = Rx_data(rd.PRNGKey(0), y_batch[num,:,None], symbWDM[num], 2, param=param, paramCh=paramCh, FO=0, lw=0)

# test data
num = 2
data_sml_ts, paramRx, noise = Rx_data(rd.PRNGKey(0), y_batch[num,:,None], symbWDM[num], 2, param=param, paramCh=paramCh, FO=0, lw=0)

print(f'data.y.shape:{data_sml_tr.y.shape}')

# Define optimizer
# lr1 = optax.piecewise_constant_schedule(1e-4,{2000:1e-5, 4000:1e-6})

# tx = optax.adam(learning_rate=lr2)
# tx = optax.adamw(learning_rate=lr2, weight_decay=0.01)
# tx = optax.chain(
#   optax.clip(1.0),
#   optax.adamw(learning_rate=schedule,weight_decay=0.0001),
# )

def fn(k,v):
    if k == 'dispersion_kernel':
        return 'D'
    elif k == 'nonlinear_kernel':
        return 'NL'
    elif 'n_bias' in k:
        return 'bias'
    else:
        return 'bias'

from flax.core.frozen_dict import FrozenDict

def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict'''
  def map_fn(nested_dict):
    return FrozenDict({k: (map_fn(v) if isinstance(v, FrozenDict) else fn(k, v))
            for k, v in nested_dict.items()})
  return map_fn

label_fn = map_nested_fn(fn)

lrD = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-4,warmup_steps=100,decay_steps=4000,end_value=1e-6)
lrN = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-4,warmup_steps=100,decay_steps=4000,end_value=1e-6)
lrO = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3,warmup_steps=100,decay_steps=4000,end_value=1e-6)
# tx = optax.multi_transform({'D':optax.adam(learning_rate=lrD), 'NL':optax.adam(learning_rate=lrN),'bias':optax.adam(learning_rate=lrO)}, label_fn)
tx = optax.multi_transform({'D':optax.adam(learning_rate=lrD), 'NL':optax.adam(learning_rate=lrN),'bias':optax.adam(learning_rate=0)}, label_fn)

## define model
batch_size = 100
sparams_flatkeys = [] #  [('DBP',),('RConv',), ('final layer',)]
steps = args.steps
k = int(1600*(25/steps))
dtaps =  k - (k%4) + 1          # 2k+1  or 4k+1
ntaps = int(601*5 / steps)
ntaps = ntaps - (ntaps%4) + 1   # 4k+1
rtaps = 41 
xi = 0
NL = True # use nonlinear layer or not.
model_train = base.model_init(data_sml_tr, init_len=60000, sparams_flatkeys=sparams_flatkeys, mode='train', steps=steps, xi=xi,  dtaps=dtaps, ntaps=ntaps, rtaps=rtaps, nonlinear_layer=NL)  
model_test = base.model_init(data_sml_ts, init_len=60000, sparams_flatkeys=sparams_flatkeys, mode='test', steps=steps, xi=xi, dtaps=dtaps, ntaps=ntaps, rtaps=rtaps, nonlinear_layer=NL)  

# Training FDBP Model on a single signal 
gen = base.train(model_train, data_sml_tr, batch_size=batch_size, n_iter=6000, tx=tx)
loss0, Train0 = base.run_result(gen)
print('Final train loss:',loss0[-1])



# test model
metric1,sig_list, l1 = base.test(model_test, Train0[-1].params, data_sml_tr, eval_range=(100,-100))
metric2,sig_list, l2 = base.test(model_test, Train0[-1].params, data_sml_ts, eval_range=(100,-100))

# train BER
print(f'\n training loss: {l1} \n')
print(metric1)

# test BER
print(f'\n testing loss: {l2} \n')
print(metric2)

tl = []
for i in range(60):
    metric2,sig_list, l2 = base.test(model_test, Train0[i].params, data_sml_ts, eval_range=(100,-100))
    tl.append(l2)
    

np.save(f'loading/train_loss', loss0)
np.save(f'loading/test_loss', tl)

# pickle.dump(Train0[-1].params, open(f'loading/params_power{args.power}_steps{args.steps}','wb'))
# np.save(f'loading/metric_train_power{args.power}_steps{args.steps}', metric1)
# np.save(f'loading/metric_test_power{args.power}_steps{args.steps}', metric2)