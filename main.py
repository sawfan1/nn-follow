# follow along for neural nine video https://www.youtube.com/watch?v=MSrfaI1gGjI

import random
import gym
import numpy as np

env = gym.make('Taxi-v3')
alpha = 0.9 # learning rate