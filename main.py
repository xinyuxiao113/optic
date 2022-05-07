## 测试进度条
from copyreg import pickle
from optical_flax.generate_data import Tx_data, channel, Rx_data
import jax
import jax.random as rd
import os
import pickle


N = [7, 13, 19, 25]
SpS = [16, 24, 32, 40]
Power = 0
Nbits = 400000
data_path = '/home/xiaoxinyu/data/0507'

batch = 10
dz = 0.5
rx_sps = 8

key_tx = rd.split(rd.PRNGKey(123), len(N))
key_rx = rd.plit(rd.PRNGKey(1234), len(N))


if not os.path.exists(data_path):
    os.mkdir(data_path)

for i in range(len(N)):
    n = N[i]
    print(f'channel {n}:')
    tx_path= data_path + f'/Tx_ch{n}'

    ## Step 1: Tx
    if os.path.exists(tx_path):
        print('Tx data has been generated before !')
        with open(tx_path, 'rb') as file:
            sigWDM, symbWDM, param = pickle.load(file)
    else:
        sigWDM, symbWDM, param = Tx_data(key_tx[i], batch, Nch=N[i], Power=Power, SpS=SpS[i])
        with open(tx_path, 'wb') as file:
            pickle.dump((sigWDM, symbWDM, param), file)

    ## Step 2: channel
    Fs = param.Rs*param.SpS  # sample rates
    sigWDM_rx, paramCh = channel(sigWDM, Fs, dz=dz)
    channel_path = data_path + f'Channel_ch{n}'
    with open(channel_path, 'wb') as file:
        pickle.dump((sigWDM_rx, paramCh), file)

    ## Step 3:
    rx_path = data_path + f'data_ch{n}'
    data_sml, paramRx = Rx_data(key_rx[i], sigWDM_rx, symbWDM, rx_sps, param=param, paramCh=paramCh)
    with open(rx_path, 'wb') as file:
        pickle.dump((data_sml, paramRx), file)
    
