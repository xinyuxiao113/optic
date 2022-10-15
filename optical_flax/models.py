from typing import Callable, Any, Optional
from flax import linen as nn
from flax import struct
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial

from optical_flax.attention import SelfAttention
from optical_flax.functions import cleaky_relu
from optical_flax.initializers import zeros, near_zeros
from optical_flax.operator import frame


c_act = cleaky_relu
r_act = jax.nn.leaky_relu
def act(x):
  if x.dtype == jnp.float32:
    return r_act(x)
  else:
    return c_act(x)



@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  dtype: Any = jnp.complex64
  param_dtype:Any = jnp.complex64
  emb_dim: int = 2
  num_heads: int = 2
  num_layers: int = 3
  qkv_dim: int = 2
  mlp_dim: int = 16
  dropout_rate: float = 0.3
  attention_dropout_rate: float = 0.3
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)



class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs, deterministic=True):
    """Applies Transformer MlpBlock module."""
    config = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(
        config.mlp_dim,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)(
            inputs)
    
    x = act(x)
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)(
            x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, deterministic):
    """Applies Encoder1DBlock module.
    Args:
      inputs: input data.
      deterministic: if true dropout is applied otherwise not.
    Returns:
      output after transformer encoder block.
    """
    config = self.config

    # Attention block.
    # assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=config.dtype,param_dtype=config.param_dtype)(inputs)
    x = SelfAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=deterministic)(
            x)

    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=config.dtype,param_dtype=config.param_dtype)(x)
    y = MlpBlock(config=config)(y, deterministic=deterministic)
    return x + y

xi_config = TransformerConfig(dtype=jnp.float32, param_dtype=jnp.float32)
H_config = TransformerConfig()

class Transformer(nn.Module):
  """Transformer Model for sequence tagging."""
  features:int=20
  config: TransformerConfig = xi_config
  nn_mode: bool=False
  result_init:Callable=zeros

  @nn.compact
  def __call__(self,inputs, train):
    """Applies Transformer model on the inputs.
    Args:
      inputs: input data
      train: if it is training.
    Returns:
      output of a transformer encoder.
    """
    # assert inputs.ndim == 3  # (batch, len, features)
    if self.nn_mode==True:   
      p = self.param('nn_weight', self.result_init, (self.features,), self.config.param_dtype)
      return jnp.stack([p,p],axis=-1)
    else:
      config = self.config
      x = inputs
      # TODO: Add embbedings.
      # x = AddPositionEmbs(config)(x)

      for _ in range(config.num_layers):
        x = Encoder1DBlock(config)(x, deterministic=not train)

      return x


class cnn_block(nn.Module):
  kernel_shapes: tuple=(3,5,3,5)  # (3,3,3)
  channels: tuple=(4,8,4,2)       # (2,2,2)
  dtype:Any = jnp.complex64
  param_dtype:Any = jnp.complex64
  '''
  Input:
  [B, L, 2]  --> [B,L,4] --> [B,L,8] --> []
  '''

  @nn.compact
  def __call__(self,inputs):
    assert len(self.kernel_shapes) == len(self.channels)
    Conv = partial(nn.Conv, strides=(1,), param_dtype=self.param_dtype, dtype=self.dtype, padding='same')
    x = inputs
    for k in range(len(self.kernel_shapes)):
      x = Conv(features=self.channels[k], kernel_size=(self.kernel_shapes[k],))(x)
      x = act(x)
    x = Conv(features=self.channels[-1], kernel_size=(1,))(inputs) + x
    return x



class CNN(nn.Module):
  features:int=1
  result_init:Callable=zeros
  depth:int=3
  block_kernel_shapes: tuple=(3,5,3,5)  # (3,3,3)
  block_channels: tuple=(4,8,4,2)       # (2,2,2)
  dtype:Any = jnp.complex64
  param_dtype:Any = jnp.complex64
  nn_mode: bool=False

  '''
    (Batch, L, channels)  --> (Batch, L, channels)
    (L, channels)  --> (L, channels)
  '''


  @nn.compact
  def __call__(self, inputs, train=False):
    if self.nn_mode==True:   
      p = self.param('nn_weight', self.result_init, (self.features,), self.param_dtype)
      return jnp.stack([p,p],axis=-1)
    else:
      x = inputs
      block = partial(cnn_block, 
               kernel_shapes=self.block_kernel_shapes,
               channels=self.block_channels, 
               dtype=self.dtype, 
               param_dtype=self.param_dtype)
      for k in range(self.depth):
        x = block()(x)
      bias = self.result_init('key',(self.features,), self.param_dtype)
      if self.features == 1:
        return jnp.sum(x, axis=0)[None,:] + bias
      else:
        return x + bias[:,None]




def embed(inputs,k, sps,mode='wrap'):
  '''
  inputs: [L*sps,2]
  Output: [L, 2*sps*(2k+1)]
  '''
  x = jnp.pad(inputs, ((k*sps,k*sps),(0,0)), mode=mode)
  x = frame(x, (2*k+1)*sps, sps)
  x = x.reshape(x.shape[0], -1) 
  return x


class Embedding(nn.Module):
  k:int=1   # additional mimo symbols
  sps:int=8
  @nn.compact
  def __call__(self, inputs):
    # inputs: (B, L*sps,2)
    if inputs.ndim == 2:
      x = embed(inputs, self.k, self.sps)
      return x
    elif inputs.ndim == 3:
      x = jax.vmap(embed, in_axes=(0,None,None), out_axes=0)(inputs, self.k, self.sps)
      return x
    else:
      raise(ValueError)
