import jax

def complex_F(f):
    def _f(x):
        y = f(x.real) + f(x.imag)*(1j)
        return y
    return _f

crelu = complex_F(jax.nn.relu)
cleaky_relu = complex_F(jax.nn.leaky_relu)

ctanh = complex_F(jax.nn.tanh)
csigmoid = complex_F(jax.nn.sigmoid)

