import jax
import jax.random as rd
import jax.numpy as jnp
import numpy as np
import os
import pickle
from tqdm import tqdm
import time
from optical_flax.fiber_system import Rx_data, get_data
from optical_flax.utils import calc_time



def Rx(key, path, sps, FO, lw):
    with open(path+'/Tx_ch7_', 'rb') as file:
        sigWDM_tx, symbWDM, param = pickle.load(file)

    with open(path+'/Channel_ch7', 'rb') as file:
        sigWDM, paramCh = pickle.load(file)

    data, paramRx, noise = Rx_data(key, sigWDM, symbWDM, sps, param, paramCh, FO, lw)
    return data, noise



def batch_data(data, Nlen=2000, Nstep=1000):
    '''
    get batched data.
    Input:
        data
    '''
    from commplax.xop import frame
    sps = data.a['sps']
    y = jax.vmap(frame, in_axes=(0,None,None), out_axes=0)(data.y, Nlen*sps, Nstep*sps).reshape([-1,Nlen*sps,2])
    x = jax.vmap(frame, in_axes=(0,None,None), out_axes=0)(data.x, Nlen, Nstep).reshape([-1,Nlen,2])
    return jax.device_get(y),jax.device_get(x)


def take_data(path, sps, Nlen, Nstep):
    '''
        take batched data from path.
    '''
    
    data_sml = get_data(path,sps,batch=True)
    y,x = batch_data(data_sml, Nlen, Nstep)
    return y, x


@calc_time
def Generation_DataSet(path_list, sps, Nlen, Nstep):
    y_lis = []
    x_lis = []
    for f in tqdm(path_list, desc='loading data'):
        y,x = take_data(f, sps, Nlen, Nstep)
        y_lis.append(y)
        x_lis.append(x)
    y_data = np.concatenate(y_lis, axis=0)
    x_data = np.concatenate(x_lis, axis=0)
    return x_data, y_data


def DataLoader(x_data, y_data, batch, rng):
    N = x_data.shape[0]
    steps = N//batch

    perms = jax.random.permutation(rng, N)
    perms = perms[:steps * batch]
    perms = perms.reshape((steps, batch))
    for perm in perms:
        yield x_data[perm],y_data[perm]
