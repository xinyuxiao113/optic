#!/bin/bash
#SBATCH -J test2            # Job名
#SBATCH -o ./out/test2.out  # 输出, 目录./out必须存在，否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=short        # qos(quality of service): normal or short or debug, 对应不同优先级及最大可用时长.
#SBATCH -p RTX3090            # 指定partition: geforce,k80,etc.
#SBATCH --cpus-per-task=1  # 申请 cpu core 数; 可用内存与申请 cpu core 数成正比.
#SBATCH --mem=40G          # 申请10G内存
#SBATCH --gres=gpu:1       # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 10:00:00       # 申请Job运行时长0小时5分钟0秒,若要申请超过一天时间,如申请1天,书写格式为#SBATCH -t 1-00:00:00

# 上述 SBATCH 参数不指定时均有系统指定的默认值

# 随着 Job 的提交和执行，slurm 会帮助用户在申请的节点上挨个执行下述命令
module load spack

module add cuda-11.4.2-gcc-11.2.0-rxy4qhm            # 载入 CUDA 9.0 模块
module add cudnn-8.2.4.15-11.4-gcc-11.2.0-a6q32ad
module add gcc/11.2.0  
module add anaconda           # 载入 anaconda 模块

source ~/.bashrc
conda activate commplax


python test.py --steps 7 --power 6.0
python test.py --steps 7 --power 3.0
python test.py --steps 7 --power 0.0
python test.py --steps 7 --power -3.0

python test.py --steps 8 --power 6.0
python test.py --steps 8 --power 3.0
python test.py --steps 8 --power 0.0
python test.py --steps 8 --power -3.0

python test.py --steps 10 --power 6.0
python test.py --steps 10 --power 3.0
python test.py --steps 10 --power 0.0
python test.py --steps 10 --power -3.0

python test.py --steps 20 --power 6.0
python test.py --steps 20 --power 3.0
python test.py --steps 20 --power 0.0
python test.py --steps 20 --power -3.0