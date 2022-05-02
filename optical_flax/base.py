import jax
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
from collections import namedtuple
from typing import Any, NamedTuple,Callable, Iterable, Optional, Tuple, Union,Sequence
from functools import partial

from tqdm.auto import tqdm
import gdbp.gdbp_base as gb
from commplax.module import core
from commplax import util, cxopt
from jax import device_get, device_put


## 相对导入
from optical_flax.initializers import fdbp_init
from optical_flax.layers import DSP_Model
from optical_flax.utils import realize

Optimizer = Any

Model = namedtuple('Model', 'module initvar overlaps name')


## Define Model
def model_init(data, 
        init_len:int=8000, 
        name='GDBP',
        sparams_flatkeys = [('FDBP',)],
        steps=3,xi=1.1, domain='time',mode='train',
        **kwargs):
    '''
    init a model
    data: data set
    init_len: length of sequence for initializing the model
    steps: DBP steps
    xi: Nonlinar parameter scale
    domain: 'time' or 'frequency'  used to get DBP initial parameters
    '''
    init_fn = fdbp_init(data.a, xi=xi, steps=steps,domain=domain)  
    model = DSP_Model(init_fn=init_fn, w0=data.w0, steps=steps,mode=mode, **kwargs)
    model = realize(model)
    y0 = core.Signal(data.y[:init_len])
    key = random.PRNGKey(0)
    z0,v0 = model.init_with_output(key,y0)
    if mode=='test':
        z0 = z0[-1]
    ol = z0.t.start - z0.t.stop
    sparams, params = util.dict_split(v0['params'],sparams_flatkeys)
    state = {'af_state':v0['af_state'], 'norm': v0['norm']}
    aux = v0['aux_inputs']
    const = v0['const']

    return Model(model, (params, state, aux, const, sparams), ol, name)


import flax
Array = Any
Dict = Union[dict, flax.core.FrozenDict]

def loss_fn(module: nn.Module,
            params: dict,
            state: dict,
            y: Array,
            x: Array,
            aux: Dict,
            const: Dict,
            sparams: Dict):
    ''' loss function

        Args:
            module: module returned by `model_init`
            params: trainable parameters
            state: module state
            y: transmitted waveforms
            x: aligned sent symbols
            aux: auxiliary input
            const: contants (internal info generated by model)
            sparams: static parameters

        Return:
            loss, updated module state
    '''

    params = util.dict_merge(params, sparams)
    z, updated_state = module.apply(
        {
            'params': params,
            'aux_inputs': aux,
            'const': const,
            **state
        }, core.Signal(y), mutable={'af_state', 'norm'}) 
    loss = jnp.mean(jnp.abs(z.val - x[z.t.start:z.t.stop])**2)  # 因为卷积valid模式会丢失一些边缘信息，最后对齐的只有x中间一段信息
    return loss, updated_state


import optax
TrainState = namedtuple('TrainState', ['params','opt_state','state'])

@partial(jax.jit, static_argnums=(0,1,-1))
def update_state(net, tx, train_state, y, x, aux, const, sparams, renew_state=True):
    aux = core.dict_replace(aux, {'truth': x})  # 将真实的x输入
    (loss, new_state), grads =jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(net, train_state.params, train_state.state, y, x, aux, const, sparams)
    updates, opt_state = tx.update(grads, train_state.opt_state)
    params = optax.apply_updates(train_state.params, updates)
    train_state = TrainState(params=params, opt_state=opt_state, state=new_state)
    return loss, train_state

from gdbp import data as gdat


def train(model: Model,
          data: gdat.Input,
          batch_size: int = 500,
          n_iter = None,
          tx: Optimizer = optax.adam(learning_rate=1e-4),
          renew_state=True):
    ''' training process (1 epoch)

        Args:
            model: Model namedtuple return by `model_init`
            data: dataset
            batch_size: batch size
            opt: optimizer

        Returns:
            yield loss, trained parameters, module state
    '''
    ## 定义trainstate
    params, state, aux, const, sparams = model.initvar
    opt_state = tx.init(params)
    train_state = TrainState(params=params, opt_state=opt_state, state=state)

    ## data loader
    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    ## training
    for i, (y, x) in tqdm(enumerate(batch_gen),total=n_iter, desc='training', leave=False):
        if i >= n_iter: break

        ## 代入数据 x (把字典中每个 truth 都替换为 x)
        aux = core.dict_replace(aux, {'truth': x}) 
        loss, train_state= update_state(model.module, tx, train_state, y, x, aux, const, sparams, renew_state)
        yield device_get(loss), train_state


from commplax import comm
def test(model: Model,
         params: Dict,
         data: gdat.Input,
         eval_range: tuple=(300000, -20000),
         metric_fn=comm.qamqot):
    ''' testing, a simple forward pass

        Args:
            model: Model namedtuple return by `model_init`
        data: dataset
        eval_range: interval which QoT is evaluated in, assure proper eval of steady-state performance
        metric_fn: matric function, comm.snrstat for global & local SNR performance, comm.qamqot for
            BER, Q, SER and more metrics.

        Returns:
            evaluated matrics and equalized symbols
    '''

    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
      params = model.initvar[0]

    res, _ = model.module.apply({
                   'params': util.dict_merge(params, sparams),
                   'aux_inputs': aux,
                   'const': const,
                   **state
               }, core.Signal(data.y), mutable={'af_state','norm'})
    z = res[-1]
    metric = metric_fn(z.val,
                       data.x[z.t.start:z.t.stop],
                       scale=jnp.sqrt(10),
                       eval_range=eval_range)
    return metric, res


def test_meta(model: Model,
         params: Dict,
         data: gdat.Input,
         eval_range: tuple=(0,-1),
         metric_fn=comm.qamqot,
         batch_size=500,
         n_iter = None):
    ''' testing, a simple forward pass

        Args:
            model: Model namedtuple return by `model_init`
        data: dataset
        eval_range: interval which QoT is evaluated in, assure proper eval of steady-state performance
        metric_fn: matric function, comm.snrstat for global & local SNR performance, comm.qamqot for
            BER, Q, SER and more metrics.

        Returns:
            evaluated matrics and equalized symbols
    '''

    ## test procedure
    state, aux, const, sparams = model.initvar[1:]
    if params is None:
      params = model.initvar[0]

    ## data loader
    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    res_list = []
    metric_list = []
    for i, (y, x) in tqdm(enumerate(batch_gen),total=n_iter, desc='Testing', leave=False):
        if i >= n_iter: break

        ## 代入数据 x (把字典中每个 truth 都替换为 x)
        aux = core.dict_replace(aux, {'truth': x}) 
        res, state = partial(jax.jit,backend='cpu',static_argnames=['mutable'])(model.module.apply)({
                   'params': util.dict_merge(params, sparams),
                   'aux_inputs': aux,
                   'const': const,
                   **state
               }, core.Signal(y), mutable=('af_state','norm'))
        
        z = res[-1]
        metric = metric_fn(z.val,
                        x[z.t.start:z.t.stop],
                        scale=jnp.sqrt(10),
                        eval_range=eval_range)
        res_list.append(res)
        metric_list.append(metric)
    
    metric = metric*0 + np.mean(metric_list, axis=0)
    
    return metric, metric_list, res_list


def run_result(generator):
    '''
    Training the model
    '''
    return list(zip(*list(generator)))


from commplax import op
def get_train_batch(ds: gdat.Input,
                    batchsize: int,
                    overlaps: int,
                    sps: int = 2):
    ''' generate overlapped batch input for training

        Args:
            ds: dataset
            batchsize: batch size in symbol unit
            overlaps: overlaps in symbol unit
            sps: samples per symbol

        Returns:
            number of symbols,
            zipped batched triplet input: (recv, sent, fomul)
    '''

    flen = batchsize + overlaps # 用 995 个符号来预测 中间 500 个符号
    fstep = batchsize 
    ds_y = op.frame_gen(ds.y, flen * sps, fstep * sps) # 接受信号y的采样率 x2
    ds_x = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0] # 查看batch的个数
    return n_batches, zip(ds_y, ds_x)

import matplotlib.pyplot as plt
from optical_flax.dsp import time_recovery_vmap

def show_symb(sig, symb, name, idx1, idx2, size=10, fig_size=(15,4), time_recovery = True):

    ## constellation
    symb_set = set(symb[:,0])

    sig_ = sig.val[::sig.t.sps][idx1]
    symb_ = symb[sig.t.start//sig.t.sps: symb.shape[0] + sig.t.stop//sig.t.sps][idx1]
    if time_recovery:
        sig_ =  time_recovery_vmap(sig_, symb_)[0]
    modes = symb_.shape[1]
    

    fig, ax = plt.subplots(1,4, figsize=fig_size)
    fig.suptitle(name)

    for sym in symb_set:
        for j in range(modes):
            sigj = sig_[:,j]
            symbj = symb_[:,j]
            z = sigj[symbj == sym]

            ax[j].scatter(z.real, z.imag, s=size)

    ## angle error, t的单位是 symbol period
    sig_ = sig.val[::sig.t.sps][idx2]
    symb_ = symb[sig.t.start//sig.t.sps: symb.shape[0] + sig.t.stop//sig.t.sps][idx2]
    if time_recovery:
        sig_ =  time_recovery_vmap(sig_, symb_)[0]
    modes = sig_.shape[1]

    for j in range(modes):
        sigj = sig_[:,j]
        symbj = symb_[:,j]
        ax[2+j].plot(np.angle(sigj/symbj))




import numpy as np
def show_fig(sig_list, symb, name, idx1=None,idx2=None,point_size=10, fig_size=(15,4)):
    if idx1 == None:
        idx1 = np.arange(symb.shape[0]//3, symb.shape[0]//3 + 10000)
    if idx2 == None:
        idx2 = np.arange(symb.shape[0]//3, symb.shape[0]//3 + 600)
    
    for i in range(len(sig_list)):
        show_symb(sig_list[i], symb, name[i], idx1,idx2,size=point_size, fig_size=fig_size)

    

