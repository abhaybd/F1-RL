import numpy as np

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
