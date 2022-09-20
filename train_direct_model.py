import pickle, re
from typing import Callable
import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn
from functools import partial
from DataLoader import data_sml,y_example, x_data, y_data,DataLoader,take_data,test_path_list, Nlen, Nstep,sps
from optical_flax.layers import MetaSSFM, GRU_DBP, fdbp, NNSSFM,Dense_net
from optical_flax.models import Transformer, CNN, TransformerConfig, Embedding
from optical_flax.initializers import fdbp_init, near_zeros,ones
from optical_flax.utils import realize,show_tree
from commplax.module.core import Signal, SigTime


######################## STEP (1) setting parameters ########################

k = 2   # additional mimo symbols
assert (Nlen-Nstep)//2-k >= 0
steps = 3
save_param = False
Epochs = 10
batch_size = 100

pmodes=2
d_init, n_init = fdbp_init(data_sml.a, xi=1.1, steps=steps, domain='frequency')
dtaps = Nlen*sps
ntaps = Nlen*sps
init_rng = rd.PRNGKey(1213)
dropout_rng = rd.PRNGKey(233)

######################## STEP (2) Define model. ########################
d_model= (2*k+1)*sps*pmodes
config = TransformerConfig(emb_dim=d_model, num_heads=pmodes, qkv_dim=pmodes*sps,mlp_dim=24)

trans = partial(Transformer, config=config, nn_mode=False)
cnn = partial(CNN,dtype=jnp.float32,
            param_dtype=jnp.float32, 
            block_kernel_shapes=(3,5,3),
            block_channels=(2*d_model, 4*d_model, d_model),
            nn_mode=False)
# dense = partial(Dense_net, width=(20,20,20),dtype=jnp.float32, param_dtype=jnp.float32,nn_mode=True)
# dense need to vmap.

class MyDBP(nn.Module):
    encoder:Callable=cnn
    discard:int=(Nlen-Nstep)//2
    @nn.compact
    def __call__(self, signal, train):
        x, t = signal
        x = Embedding(k=k, sps=sps)(x) 
        x = self.encoder()(x) 
        # x [B,L,C]
        x = nn.Conv(features=2,kernel_size=(1,),strides=(1,), param_dtype=jnp.complex64,dtype=jnp.complex64, padding='valid')(x)
        x = x[:,self.discard:x.shape[1]-self.discard:,:]
        t = SigTime(self.discard,-self.discard,1)
        return Signal(x,t)
    
net = MyDBP()
LDBP = realize(net)
var0 = LDBP.init(init_rng, Signal(y_example), False)
print(show_tree(var0))

######################## STEP (3) Define optimizer. ########################

schedule = optax.warmup_cosine_decay_schedule(
  init_value=0.0,
  peak_value=0.1,
  warmup_steps=10,
  decay_steps=100,
  end_value=0.001,
)

tx = optax.chain(
  optax.clip(1.0),
  optax.adamw(learning_rate=schedule,weight_decay=0.0001),
)
# tx = optax.adam(learning_rate=1e-3)

######################## STEP (4) define loss and update step.########################

def train_loss(var, xi, yi, key): 
    ## TODO: determine the dropout rate and train mode. FIXME:
    x = LDBP.apply(var, Signal(yi), True, rngs={'dropout':key})
    s = (Nlen - Nstep)//2
    e = s + Nstep
    return jnp.mean(jnp.abs(x.val - xi[:,s:e,:])**2)


def test_loss(var, xi, yi): 
    x = LDBP.apply(var, Signal(yi), False)
    s = (Nlen - Nstep)//2
    e = s + Nstep
    return jnp.mean(jnp.abs(x.val - xi[:,s:e,:])**2)


@jax.jit
def update_param(var, opt_state, xi, yi, key):
    loss_val, grads = jax.value_and_grad(train_loss)(var, xi,yi, key)
    updates, opt_state = tx.update(grads, opt_state, var)
    var = optax.apply_updates(var, updates)
    return var, opt_state, loss_val


######################## STEP (5) Training model. ########################

loss_dict = {}
loss_dict['train'] = []
for test_path in test_path_list:
    loss_dict[test_path] = []
opt_state0 = tx.init(var0)



for t in range(Epochs):
    
    DL = DataLoader(x_data,y_data,batch_size,jax.random.PRNGKey(0))
    for i,(xi,yi) in enumerate(DL):
        key = rd.fold_in(dropout_rng, t)
        var0, opt_state0, l = update_param(var0, opt_state0, xi, yi, key)
        loss_dict['train'].append(l)
        print(f'Epoch {t} -- batch {i} -- train loss {l}')
    
    for test_path in test_path_list:
        yi,xi = take_data(test_path, sps, Nlen, Nstep)
        l = test_loss(var0, xi, yi)
        loss_dict[test_path].append(l)

        i = re.search('power',test_path).start(0)
        print(f'########### Epoch {t} -- Test loss on {test_path[i:i+8]}: {l}  ##############')
    if save_param:
        with open(f'loading/param_NN/state_epoch{t}', 'wb') as file:
            pickle.dump({'param':var0, 'loss_val':l}, file)
    


with open('loading/loss_0617_Meta_ntaps1','wb') as file:
    pickle.dump(loss_dict, file)