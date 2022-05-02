## 测试进度条
from tqdm import tqdm
from time import sleep
from optical_flax.generate_data import Tx_data, Rx_data
import os
from optical_flax.utils import MSE
import jax
from jax.random import PRNGKey as Key, split


N = [7, 13, 19, 25]
SpS = [16, 24, 32, 40]
Power = 0
Nbits = 400000
data_path = 'data/data0426_N4e5'

batch = 1
dz = 0.5
rx_sps = 8

key_tx = split(Key(123), len(N))
key_rx = split(Key(1234), len(N))


if not os.path.exists(data_path):
    os.mkdir(data_path)

for i in range(len(N)):
    n = N[i]
    print(f'channel {n}:')
    tx_path= data_path + f'/Tx_ch{n}'
    if os.path.exists(tx_path):
        print('Tx data has been generated before !')
        pass
    else:
        Tx_data(key_tx[i], batch, Nch=n, SpS=SpS[i], Power=Power, Nbits=Nbits, path = tx_path)
    Rx_data(key_rx[i], tx_data_path=tx_path, rx_data_path = data_path + f'/dataset_ch{n}_dz{dz}_N4e5', dz=dz, sps=rx_sps)
