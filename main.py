import warnings
warnings.filterwarnings('ignore')

from ddpg import DDPG
from a2c import A2C
from ppo import PPO
from td3 import TD3
import plot
import torch.multiprocessing as mp
import os


def main():

    plot.initialize()
    mp.set_start_method('spawn')

    for i in range(5):
        print(f"---------- round {i} ----------")

       
            
        if not os.path.isfile(f'plots/sac/{i}2_testing.png'):
            ddpg = DDPG(state_type='indicators', djia_year=2014, repeat=i)
            ddpg.train()
            ddpg.test()


if __name__ == '__main__':
    main()
