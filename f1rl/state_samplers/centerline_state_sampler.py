import numpy as np
from ..env_wrapper import F1EnvWrapper
from f110_gym.envs import F110Env
from ..reward_fns.speed_reward import get_reward
from ..state_featurizers.lidar_and_curvature import transform_state

def create_state_sampler(seed=None):
    rng = np.random.default_rng(seed=seed)
    angle_delta = 15 * np.pi / 180  # 15 degrees in radians
    
    def sample_state(env):
        size = env.centerline.shape[0]
        idx = rng.choice(size)
        point = env.centerline[idx]
        next = env.centerline[(idx + 1) % size]
        angle = np.arctan2(next[1] - point[1], next[0] - point[0]) + rng.random() * 2 * angle_delta - angle_delta
        return np.array([point[0], point[1], angle]).reshape((1, 3))

    return sample_state

if __name__ == '__main__':
    sampler = create_state_sampler(seed=0)
    env = F1EnvWrapper(F110Env(map='f1rl/maps/austin', num_agents=1), sampler, transform_state, get_reward)
    print(sampler(env))