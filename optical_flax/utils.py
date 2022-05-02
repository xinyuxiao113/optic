import jax
import jax.numpy as jnp
from jax import device_put, device_get
import jax.random as random
import flax.linen as nn
from functools import partial, wraps
from typing import Any, NamedTuple
from flax.core import freeze, unfreeze
from commplax.module.core import Signal

def normal_init(key,shape, dtype = jnp.float32):
    k1,k2 = random.split(key)
    x = random.normal(k1,shape)  + 1j * random.normal(k2,shape)
    return x.astype(dtype)


def show_tree(tree):
    return jax.tree_map(lambda x:x.shape, tree)

def c2r(x):
    '''
    [shape] --> [2,shape]
    '''
    if (x.dtype == jnp.complex64) or (x.dtype == jnp.complex128):
        return jnp.array([x.real, x.imag])
    else:
        return jnp.array([x])

def r2c(x):
    '''
    x: [2,shape] --> [shape]
    '''
    if x.shape[0] == 2:
        return x[0] + (1j)*x[1]
    else:
        return x[0]

def tree_c2r(var, key='params'):
    '''
    把 var 中的 params 变成实参数
    '''
    var = unfreeze(var)
    var[key] = jax.tree_map(c2r,var[key])
    return freeze(var)

def tree_r2c(var, key='params'):
    '''
    把 var 中的 params 变成复参数
    '''
    var = unfreeze(var)
    var[key] = jax.tree_map(r2c,var[key])
    return freeze(var)

class realModel(NamedTuple):
    init: Any
    apply: Any
    init_with_output: Any

def realize(model):
    @wraps(model.init)
    def _init(*args, **kwargs):
        var =  model.init( *args, **kwargs)
        return tree_c2r(var)
    
    @wraps(model.apply)
    def _apply(var_real, *args, **kwargs):
        var = tree_r2c(var_real)
        out = model.apply(var, *args, **kwargs)
        return out
    
    @wraps(model.init_with_output)
    def _init_with_output(key, *args, **kwargs):
        z,v = model.init_with_output(key, *args, **kwargs)
        return z, tree_c2r(v)
    
    return realModel(init=_init,apply=_apply, init_with_output=_init_with_output)



## nn_vmap
def nn_vmap_signal(module):
    '''
    将 net vmap 到 Signal第一个分量的axis=-1上, 并且参数不共享. 
    输入为Signal
    '''
    return nn.vmap(module, 
    variable_axes={'params':-1, 'const':None},  # 表示变量'params'会沿着axis=-1复制, 'const'不会复制
    split_rngs={'params':True}, # 表示初始化的种子会split
    in_axes=(Signal(-1, None),), out_axes=Signal(-1, None)) # 标记输入和输出的vmap轴, None表示vmap不作用到这个轴上


def nn_vmap_x(module):
    '''
    将 net vmap 到 Signal第一个分量的axis=-1上, 并且参数不共享. 
    输入为 Array
    '''
    return nn.vmap(module, 
    variable_axes={'params':-1},  # 表示变量'params'会沿着axis=-1复制, 'const'不会复制
    split_rngs={'params':True}, # 表示初始化的种子会split
    in_axes=-1, out_axes=-1) 



def MSE(y,y1):
    return jnp.sum(jnp.abs(y-y1)**2)



def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    return jnp.fft.ifft( jnp.fft.fft(signal)*jnp.fft.fft(ker) )

def corr_circ(x, y):  # 不交换
    '''
    x: fixed
    z[n] = x[i] y[i - n] = x[i + n]y[i]
    '''
    return conv_circ(x, jnp.roll(jnp.flip(y),1))


def auto_rho(x,y):
    '''
        计算循环 相关系数
    '''
    N = len(x)
    Ex = jnp.mean(x)
    Ey = jnp.mean(y)
    Vx = jnp.var(x)
    Vy = jnp.var(y)
    return (corr_circ(x,y)/N - Ex*Ey)/jnp.sqrt(Vx*Vy)