import jax
import jax.numpy as jnp
import flax.linen as nn

def complex_F(f):
    def _f(x):
        y = f(x.real) + f(x.imag)*(1j)
        return y
    return _f

crelu = complex_F(jax.nn.relu)
cleaky_relu = complex_F(jax.nn.leaky_relu)

ctanh = complex_F(jax.nn.tanh)
csigmoid = complex_F(jax.nn.sigmoid)



class modeReLU(nn.Module):

    @nn.compact
    def __call__(self, x):
        b = self.param('b', lambda *_:-jnp.array(0.01))
        return jax.nn.relu(jnp.abs(x)+b)*x/jnp.abs(x)

relu = jax.nn.relu

def zReLU(z):
    return (z.real > 0)*(z.imag > 0) * z