import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import flax.linen as nn
from functools import partial, wraps
from typing import Any, NamedTuple,Callable, Iterable, Optional, Tuple, Union,Sequence
from collections import namedtuple
from flax.core import freeze, unfreeze,lift
from commplax.module.core import SigTime, Signal, zeros, conv1d_t, vmap,wpartial,delta,gauss
from commplax import xop, xcomm
from commplax import adaptive_filter as af
from commplax.module import core
from commplax import cxopt

import sys
sys.path.append('/home/xiaoxinyu/optic')

from optical_flax.functions import crelu, ctanh, csigmoid
from optical_flax.initializers import near_zeros, ones
from optical_flax.utils import nn_vmap_x, nn_vmap_signal, normal_init


############################################## Adaptivefilter ##################################################

## BPN Module
class BPN(nn.Module):
    momentum: float = 0.999
    mode: str = 'train'  # or 'test'
    
    @nn.compact
    def __call__(self,signal):
        x = signal.val
        running_mean = self.variable('norm', 'running_mean',lambda *_: 0. + jnp.ones(x.shape[-1]), ())
        if self.mode == 'train':
            mean = jnp.mean(jnp.abs(x)**2, axis=0)
            running_mean.value = self.momentum * running_mean.value + (1 - self.momentum) * mean
        else:
            mean = running_mean.value
        return signal / jnp.sqrt(mean)


## signal conv1d
class conv1d(nn.Module):
    taps:int = 31
    rtap:Any =None
    mode:str='valid'
    kernel_init:Callable=delta ##
    conv_fn:Callable=xop.convolve

    @nn.compact
    def __call__(self,signal):
        x,t = signal
        t = self.variable('const', 't', conv1d_t, t, self.taps, self.rtap, 1, self.mode).value
        h = self.param('kernel', self.kernel_init, (self.taps,), jnp.complex64)
        x = self.conv_fn(x, h, mode=self.mode)

        return Signal(x, t)

## signal mimoconv1d
class mimoconv1d(nn.Module):
    taps:int=31
    rtap:Any=None
    dims:int=2
    mode:str='valid'
    kernel_init:Callable=zeros
    conv_fn:Callable=xop.convolve

    @nn.compact
    def __call__(self, signal):
        x, t = signal
        t = self.variable('const', 't', conv1d_t, t, self.taps, self.rtap, 1, self.mode).value
        h = self.param('kernel', self.kernel_init, (self.taps, self.dims, self.dims), jnp.float32)
        y = xcomm.mimoconv(x, h, mode=self.mode, conv=self.conv_fn)
        return Signal(y, t)


## MIMO adaptive filter
class mimoaf(nn.Module):
    taps:int=32
    rtap:Any=None
    dims:int=2
    sps:int=2
    train: Any=False
    mimofn: Any=af.ddlms
    mimokwargs:Any=freeze({})
    mimoinitargs:Any=freeze({})
    '''
    input: (x,t)   x: [2N, 2]   t: (0,0,2)
    output: (x,t)  x: [N-15, 2], t: (7,-8,1)
        [(N - taps)/2] + 1
    '''

    
    @nn.compact
    def __call__(self,signal):
        x, t = signal
        t = self.variable('const', 't', conv1d_t, t, self.taps, self.rtap, 2, 'valid').value

        # x [M, taps, 2], M = N-15
        x = xop.frame(x, self.taps, self.sps)  
        mimo_init, mimo_update, mimo_apply = self.mimofn(train=self.train, **self.mimokwargs)


        state = self.variable('af_state', 'mimoaf', lambda *_: (0, mimo_init(dims=self.dims, taps=self.taps, **self.mimoinitargs)), ())
        truth_var = self.variable('aux_inputs', 'truth',lambda *_: None, ())
        truth = truth_var.value
        if truth is not None:
            # truth: [M, 2]
            truth = truth[t.start: truth.shape[0] + t.stop]

        # af_step: int    af_stats: (w,f,s,b,fshat)   [taps,2,2] (2,) (2,) (2,) (2,)
        af_step, af_stats = state.value

        # af_step: int    af_state: (w,f,s,b,fshat) (已经利用所有输入的数据更新)
        # af_weight(Pytree): (w, f, s, b) 每个元素的axis=0为迭代轴    _: (l,d)   (当前步未及时更新的weight)
        af_step, (af_stats, (af_weights, _)) = af.iterate(mimo_update, af_step, af_stats, x, truth)

        # y: [M,2]
        y = mimo_apply(af_weights, x) # 所有的 symbol 都过了之后才运用filter
        state.value = (af_step, af_stats)
        return Signal(y, t)


## mimofoeaf
class mimofoeaf(nn.Module):
    framesize:int=100
    w0:Any=0
    train:Any=False
    preslicer:Callable=lambda x: x
    foekwargs:Any=freeze({})
    mimofn:Callable=af.rde
    mimokwargs:Any=freeze({})
    mimoinitargs:Any=freeze({})

    @nn.compact
    def __call__(self,signal):
        sps = 2
        dims = 2
        tx = signal.t  # signal.val:[1090,2]   t:(450,-450,2)
        # MIMO
        ## slisig: [1030, 2], (480,-480,2)
        slisig = self.preslicer(signal)

        ## auxsig: [500,2], (247, -248, 1)
        auxsig = mimoaf(mimofn=self.mimofn,
                        train=self.train,
                        mimokwargs=self.mimokwargs,
                        mimoinitargs=self.mimoinitargs,
                        name='MIMO4FOE')(slisig)
        # y [500, 2]  ty: (22,-23,1)
        y, ty = auxsig # assume y is continuous in time

        ## yf: [5,100,2]
        yf = xop.frame(y, self.framesize, self.framesize)

        ## af.array 复制dims份ADF分别应用于不同dim, 作用的axis=-1， CPR中不再用到 truth
        foe_init, foe_update, _ = af.array(af.frame_cpr_kf, dims)(**self.foekwargs)
        state = self.variable('af_state', 'framefoeaf', lambda *_: (0., 0, foe_init(self.w0)), ())

        # phi： float  af_step: int  af_stats: ((2, 1, 2), (2, 2, 2), (2, 2, 2)) = (z,P,Q)
        # phi:第一个符号应当旋转的角度（中间1000个）, af_stats的axis=-1为不同的极化方向 z=(theta, w)
        phi, af_step, af_stats = state.value

        # wf: [5,2] 收集了5个Block的 omega_k^-
        af_step, (af_stats, (wf, _)) = af.iterate(foe_update, af_step, af_stats, yf)  

        # wp: [5] 两个极化方向的数据几乎一致，这里直接取平均值
        wp = wf.reshape((-1, dims)).mean(axis=-1)

        # w:[1000]  framesize: 100
        w = jnp.interp(jnp.arange(y.shape[0] * sps) / sps, jnp.arange(wp.shape[0]) * self.framesize + (self.framesize - 1) / 2, wp) / sps
        # psi: [1000]
        psi = phi + jnp.cumsum(w)

        ## state value 更新
        state.value = (psi[-1], af_step, af_stats)

        # apply FOE to original input signal via linear extrapolation
        # psi_ext: [1090]
        psi_ext = jnp.concatenate([w[0] * jnp.arange(tx.start - ty.start * sps, 0) + phi,
                                psi,
                                w[-1] * jnp.arange(tx.stop - ty.stop * sps) + psi[-1]])

        signal = signal * jnp.exp(-1j * psi_ext)[:, None]
        return signal



################################################ FDBP module ################################################
class fdbp(nn.Module):
    steps:int=3
    dtaps:int=261
    ntaps:int=41
    sps:int=2
    d_init:Callable=delta
    n_init:Callable=gauss
    conv1d:Callable=conv1d
    mimoconv1d:Callable=mimoconv1d

    @nn.compact
    def __call__(self,signal):
        x, t = signal
        dconv = nn_vmap_signal(wpartial(conv1d, taps=self.dtaps, kernel_init=self.d_init))

        for i in range(self.steps):
            x, td = dconv()(Signal(x, t))
            c, t = mimoconv1d(taps=self.ntaps,kernel_init=self.n_init)(Signal(jnp.abs(x)**2, td))
            x = jnp.exp(1j * c) * x[t.start - td.start: t.stop - td.stop + x.shape[0]]
        return Signal(x, t)




## compose model
class Sequential(nn.Module):
  layers: Sequence[nn.Module]

  def __call__(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
  
class DSP_Model(nn.Module):
    steps: int = 3
    dtaps: int = 261
    ntaps: int = 41
    rtaps: int = 61
    init_fn: tuple = (core.delta, core.gauss)
    w0: Any = 0.0
    mode: str = 'train'
    GDBP: Callable = fdbp

    def setup(self):
        d_init,n_init = self.init_fn
        if self.mode == 'train':
            # configure mimo to its training mode
            mimo_train = True
        elif self.mode == 'test':
            # mimo operates at training mode for the first 200000 symbols,
            # then switches to tracking mode afterwards
            mimo_train = cxopt.piecewise_constant([200000], [True, False])
        else:
            raise ValueError('invalid mode %s' % self.mode)
            
        self.DBP = self.GDBP(name='DBP',
                steps=self.steps,  
                dtaps=self.dtaps,
                ntaps=self.ntaps,
                d_init=d_init,
                n_init=n_init)
        self.BPN = BPN(name='BPN',mode=self.mode)
        self.FOEAF = mimofoeaf(name='FOEAF',
                            w0=self.w0,
                            train=mimo_train,
                            preslicer=core.conv1d_slicer(self.rtaps),
                            foekwargs={})
        self.RConv = nn_vmap_signal(conv1d)(name='RConv', taps=self.rtaps) # vectorize column-wise Conv1D
        self.MIMOAF = mimoaf(name='MIMOAF',train=mimo_train)     # adaptive MIMO layer
        
            
    def __call__(self, signal):
        x0 = signal
        x1 = self.DBP(x0)
        x2 = self.BPN(x1)
        x3 = self.FOEAF(x2)
        x4 = self.RConv(x3)
        x5 = self.MIMOAF(x4)
        if self.mode=='train':
            return x5
        else:
            return x0, x1, x2, x3, x4, x5


################################################ Meta-SSFM module ################################################
fft = jnp.fft.fft
ifft = jnp.fft.ifft

from optical_flax.functions import cleaky_relu


class Dense_net(nn.Module):
    features:int = 20
    width:list = (60,60)
    dropout_rate:float = 0.5
    dtype:Any = jnp.complex64
    param_dtype:Any = jnp.complex64
    act:Callable=cleaky_relu
    result_init:Callable=zeros
    nn_mode: bool=False
    '''
        input: 
        case 0: [dtaps, 2] --> [dtaps,2]  (encoder)
        case 2: [dtaps, 2] --> [1, 2]  (encoder + regression)


        nn_mode: Degenerate to NNSSFM if True.
        to_real: Transform output to real number if True.
    '''

    @nn.compact
    def __call__(self, inputs, train=False):
        if self.nn_mode==True:   
            p = self.param('nn_weight', self.result_init, (self.features,), self.param_dtype)
            return jnp.stack([p,p],axis=-1)
        else:
            Dense  = partial(nn.Dense,dtype=self.dtype, param_dtype=self.param_dtype)
            x = nn.LayerNorm(dtype=self.dtype,param_dtype=self.param_dtype,reduction_axis=0,feature_axis=-1)(inputs)
            x = x.reshape(-1)
            for w in self.width:
                x = Dense(features=w)(x)
                x = self.act(x) 
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
            # TODO: weight initializers.
            # x = Dense(features=self.features, bias_init=self.result_init)(x)
            
            # FIXME: Another choise  
            bias = self.result_init('key',(self.features,), self.param_dtype)
            x = Dense(features=self.features, use_bias=False)(x) + bias
            return jnp.stack([x,x],axis=-1)


class MetaSSFM(nn.Module):
    steps:int=3
    d_init: Callable=zeros
    n_init: Callable=zeros
    dtaps: int=1024
    ntaps: int=1
    discard:int=450
    Meta_H: Callable = partial(Dense_net, width=(5,5))
    Meta_xi: Callable = partial(Dense_net, width=(5,5), dtype=jnp.float32, param_dtype=jnp.float32)


    def setup(self):
        ## 分别对 Ex, Ey 做 meta
        self.H = [self.Meta_H(features=self.dtaps, result_init=self.d_init) for i in range(self.steps)]
        # self.H = [self.param('H'+str(i), self.d_init, (self.dtaps,)) for i in range(self.steps)]
        self.phi = [self.Meta_xi(features=self.ntaps, result_init=ones) for i in range(self.steps)]
    
    def __call__(self, signal, train=False):
        x,t = signal # x : [dtaps, 2] complex
        varphi = self.n_init('key', (1,))
        for i in range(self.steps):
            x = ifft(fft(x, axis=0) * self.H[i](x, train=train), axis=0)  # H(x): [dtaps, 2]
            power = jnp.sum(jnp.abs(x)**2, axis=1)[:,None]  # power: [dtaps,1]   self.phi[i](x): [1, 2] or [dtaps, 2]
            x = x * jnp.exp(1j * varphi * self.phi[i](jnp.abs(x)**2,train=train) * power)
        t = SigTime(self.discard,-self.discard,t.sps)
        return Signal(x[self.discard:x.shape[0]-self.discard],t)      


class NNSSFM(nn.Module):
    steps:int=3
    d_init: Callable=zeros
    n_init: Callable=zeros
    dtaps: int=1024
    ntaps: int=1
    discard:int=450
    

    @nn.compact
    def __call__(self, signal):
        x,t = signal
        for i in range(self.steps):
            H = self.param('H'+str(i), self.d_init, (self.dtaps,))
            xi = self.param('xi'+str(i), self.n_init, (self.ntaps,))
            x = ifft(fft(x, axis=0) * H[:,None], axis=0)
            power = jnp.expand_dims(jnp.sum(jnp.abs(x)**2, axis=1), axis=1)
            x = x * jnp.exp(1j * xi[...,None] * power)
        t = SigTime(self.discard,-self.discard,2)
        return Signal(x[self.discard:x.shape[0]-self.discard],t)

################################################ GRU module ################################################
## GRU
class cGRU(nn.Module):
  param_dtype:Any=jnp.complex64
  dtype:Any=jnp.complex64
  kernel_init: Any=normal_init
  bias_init:Any=normal_init
  gate_fn:Any=csigmoid
  activation_fn:Any=ctanh

  @nn.compact
  def __call__(self, c, xs):
    NN = nn.scan(nn.GRUCell,
                   variable_broadcast="params",
                   split_rngs={"params": False},
                   in_axes=1,   # along axis=1 scan
                   out_axes=1)  # output on axis=1
    ## 这里可以修改 GRU的init函数
    
    return NN(param_dtype=self.param_dtype, dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            gate_fn=self.gate_fn,
            activation_fn=self.activation_fn)(c, xs)

class rGRU(nn.Module):
  param_dtype:Any=jnp.float32
  dtype:Any=jnp.float32
  kernel_init: Any=normal_init
  bias_init:Any=normal_init

  @nn.compact
  def __call__(self, c, xs):
    NN = nn.scan(nn.GRUCell,
                   variable_broadcast="params",
                   split_rngs={"params": False},
                   in_axes=1,   # along axis=1 scan
                   out_axes=1)  # output on axis=1
    ## 这里可以修改 GRU的init函数
    
    return NN(param_dtype=self.param_dtype, dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(c, xs)


frame = jax.vmap(xop.frame,in_axes=(0,None,None),out_axes=0)

class GRU_Conv(nn.Module):
    taps:int=261
    rtap:Any=None
    mode:Any='valid'
    batch_size: int = 500
    sps:int = 2
    w_init:Callable = zeros


    @nn.compact
    def __call__(self, signal):
        '''
        Input: (x,t),h
            x: batch x L x 2
            h: batch x taps x 2
            t:(int,int,int)
        Output: h, (x,t)
            h: batch x taps x 2
            x: batch x (L-dtaps+1) x 2
            t: (int,int,int)
        '''
        gru_x = cGRU(param_dtype=jnp.complex64, dtype=jnp.complex64)
        gru_y = cGRU(param_dtype=jnp.complex64, dtype=jnp.complex64)

        x,t = signal

        # 计算时间的变换 （因为卷积损失边界）
        t = self.variable('const', 't', conv1d_t, t, self.taps, self.rtap, 1, self.mode).value
        ctaps = (self.taps - 1)//2
        h = self.variable('af_state','carry',jnp.zeros, [x.shape[0], self.taps, 2], jnp.complex64)
        hx = h.value[:,:,0] # batch x taps
        hy = h.value[:,:,1] 

        hx, hs_x = gru_x(hx, x)  # hs_x : batch x L x taps
        hy, hs_y = gru_y(hy, x)  # hs_y : batch x L x taps
        xs = frame(x, self.taps, 1) # xs: batch x (L - dtaps + 1) x dtaps x 2
        Ex = jnp.sum(hs_x[:,ctaps:-ctaps,:] * xs[:,:,:,0], axis=-1) # batch x (L - dtaps + 1)
        Ey = jnp.sum(hs_y[:,ctaps:-ctaps,:] * xs[:,:,:,1], axis=-1)
        x = jnp.stack([Ex,Ey], axis=-1) # batch x (L - dtaps + 1) x 2

        hx_out = hs_x[:,self.batch_size * self.sps,:] # [batch, taps]
        hy_out = hs_y[:,self.batch_size * self.sps,:] # [batch, taps]

        ## 更新GRU hidden state
        h.value = jnp.stack([hx_out,hy_out], axis=-1) # batch x taps x 2
        return  Signal(x,t)
    

class GRU_NL(nn.Module):
    dtype:Any=jnp.float32
    taps:int=1
    batch_size: int=500
    sps:int=2
    

    @nn.compact
    def __call__(self,signal):
        '''
        Input: (x,t), h
            x: batch x L x 2
            h: batch x taps x 2     taps=1
            t:(int,int,int)
        Output: h, (x,t)
            h: batch x taps x 2
            x: batch x (L-dtaps+1) x 2
            t: (int,int,int)
        '''
        gru_x = rGRU(param_dtype=self.dtype, dtype=self.dtype)
        gru_y = rGRU(param_dtype=self.dtype, dtype=self.dtype)
    
        x,t = signal

        h = self.variable('af_state','carry',jnp.zeros, [x.shape[0], self.taps, 2], jnp.float32)
        hx = h.value[:,:,0] # batch x taps
        hy = h.value[:,:,1] 

        hx, hs_x = gru_x(hx, jnp.abs(x)**2)  # hs_x : batch x L x 1
        hy, hs_y = gru_y(hy, jnp.abs(x)**2)  # hs_y : batch x L x 1

        hx_out = hs_x[:,self.batch_size * self.sps,:] # [batch, 1]
        hy_out = hs_y[:,self.batch_size * self.sps,:] # [batch, 1]
        
        ## 更新 state
        h.value = jnp.stack([hx_out,hy_out], axis=-1) # batch x 1 x 2
        hs = jnp.concatenate([hs_x,hs_y], axis=-1) # hs:batch x L x 2
        x = x * jnp.exp((1j) * hs)
        return Signal(x,t)


from jax.lax import stop_gradient 

class GRU_DBP(nn.Module):
    steps:int=3
    d_init: Callable=zeros
    n_init: Callable=zeros
    dtaps: int=261
    ntaps: int=1
    
    @nn.compact
    def __call__(self, signal):
        L_step = partial(GRU_Conv, taps=self.dtaps)   #!ToDo 尚未指定初始条件
        NL_step = GRU_NL

        signal = Signal(signal.val[None,...], signal.t)

        for i in range(self.steps):
            signal = L_step(w_init=self.d_init)(signal)
            signal = NL_step(w_init=self.n_init)(signal)
        return Signal(signal.val[0,:,:],signal.t)
        

