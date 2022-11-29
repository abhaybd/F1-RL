import numpy as np
import gym

def create_state_sampler(env, seed=None):
    rng = np.random.default_rng(seed=seed)