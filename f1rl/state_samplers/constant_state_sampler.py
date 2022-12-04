import numpy as np

def create_state_sampler():
    def sample_state(env):
        point = env.centerline[0]
        next = env.centerline[1]
        dx, dy = next[:2] - point[:2]
        angle = np.arctan2(dy, dx)
        return np.array([point[0], point[1], angle]).reshape((1, -1))

    return sample_state
