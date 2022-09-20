## 测试进度条
from copyreg import pickle
from optical_flax.fiber_system import Tx_data, channel, Rx_data
import jax
import jax.random as rd
import os
import pickle


N = [19]
SpS = 32
Power = [-6, -3, 0, 3, 6, 9]    # train
# Power = [-10, -5, 0, 5, 10]   # test
Nbits = 400000
data_path = '/home/xiaoxinyu/data/0531train'

batch = 10
dz = 0.5
rx_sps = 8
FO = 0    
lw = 0

key_tx = rd.split(rd.PRNGKey(5217), len(N))
key_ch = rd.split(rd.PRNGKey(5318), len(N))
key_rx = rd.split(rd.PRNGKey(5419), len(N))


if not os.path.exists(data_path):
    os.mkdir(data_path)

for i in range(len(N)):
    n = N[i]
    for j in range(len(Power)):
        power = Power[j]
        print(f'     channel {n}  Power {power}      ')

        ## Step 1: Tx
        tx_path= data_path + f'/Tx_ch{n}_power{power}'
        if os.path.exists(tx_path):
            print('Tx data has been generated before !')
            with open(tx_path, 'rb') as file:
                sigWDM, symbWDM, param = pickle.load(file)
        else:
            sigWDM, symbWDM, param = Tx_data(key_tx[i], batch, Nch=N[i], Power=power, SpS=SpS, Nbits=Nbits)
            with open(tx_path, 'wb') as file:
                pickle.dump((sigWDM, symbWDM, param), file)


        ## Step 2: channel
        channel_path = data_path + f'/Channel_ch{n}_power{power}'
        if os.path.exists(channel_path):
            print('channel data has been generated before !')
            with open(channel_path, 'rb') as file:
                sigWDM_rx, paramCh= pickle.load(file)
        else:
            Fs = param.Rs*param.SpS  # sample rates
            sigWDM_rx, paramCh = channel(key_ch[i], sigWDM, Fs, dz=dz)
            with open(channel_path, 'wb') as file:
                pickle.dump((sigWDM_rx, paramCh), file)
        

        ## Step 3:
        rx_path = data_path + f'/dataset/data_ch{n}_power{power}_FO{FO}_lw{lw}'
        if os.path.exists(rx_path):
            print('Rx data has been generated before !')
        else:
            data_sml, paramRx, noise = Rx_data(key_rx[i], sigWDM_rx, symbWDM, rx_sps, param=param, paramCh=paramCh, FO=FO, lw=lw)
            with open(rx_path, 'wb') as file:
                pickle.dump((data_sml, paramRx, noise), file)
        jax.profiler.save_device_memory_profile(f"memory{j}.prof")
        
        

        
