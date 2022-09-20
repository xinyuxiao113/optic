import numpy as np
import jax
import jax.random as rd
import jax.numpy as jnp
import matplotlib.pyplot as plt
from optical_flax.utils import parameters
from optical_flax.fiber_system import Rx_data, get_data
import pickle
from functools import partial


train_path = '/home/xiaoxinyu/data/0508'
test_path = '/home/xiaoxinyu/data/0517'
key = rd.PRNGKey(233)

k=2
Nlen=2000
Nstep=1000
sps=8
steps=3
Nfft = Nlen*sps

def batch_data(data, Nlen=2000, Nstep=1000):
    from commplax.xop import frame
    sps = data.a['sps']
    y = jax.vmap(frame, in_axes=(0,None,None), out_axes=0)(data.y, Nlen*sps, Nstep*sps).reshape([-1,Nlen*sps,2])
    x = jax.vmap(frame, in_axes=(0,None,None), out_axes=0)(data.x, Nlen, Nstep).reshape([-1,Nlen,2])
    return y,x

## load data
with open('loading/data','rb') as file:
    data = pickle.load(file)
    data_train=data['data_train']
    data_test = data['data_test']
y,x = batch_data(data_train)
y_test, x_test = batch_data(data_test)

import torch
from optical_torch import NNFiber,MetaFiber

tsf = lambda x: torch.tensor(jax.device_get(x))

y = tsf(y)
x = tsf(x)
y_test = tsf(y_test)
x_test = tsf(x_test)

from optical_flax.initializers import fdbp_init
d_init, n_init = fdbp_init(data_train.a, xi=1.1, steps=steps, domain='frequency')
H = d_init(key, (Nfft,))
phi = n_init(key, (1,))
H = tsf(H).to(torch.complex64)
phi = tsf(phi)
net = MetaFiber(steps, Nfft, H, phi)
# net = NNFiber(steps, Nfft, H, phi)


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
def loss(xi,yi):
    y1 = net(yi)
    return torch.mean(torch.abs(y1[:,500:1500,:]-xi[:,500:1500,:])**2)

L1 = []
L2 = []

for k in range(20):
    for i in range(10):
        xi = x[99*i:99*(i+1),...]
        yi = y[99*i:99*(i+1),...]
        optimizer.zero_grad()
        l = loss(xi,yi)
        l.backward()
        optimizer.step()
        print(f'Epoch {k} batch {i}, train loss: {l.item()}')
        L1.append(l.item())
    xi = x_test[99*4:99*(4+1),...]
    yi = y_test[99*4:99*(4+1),...]
    l = loss(xi,yi)
    L2.append(l.item())
    print(f'Epoch {k} , test loss: {l.item()}')
