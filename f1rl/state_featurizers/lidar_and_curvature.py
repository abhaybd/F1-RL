import numpy as np

N_LIDAR = 20
N_CURVATURE_POINTS = 10
CURVATURE_LOOKAHEAD = 0.5

def transform_state(env, state):
    # TODO: implement
    state_size = 3 + 2 + 1 + N_LIDAR + 1 + N_CURVATURE_POINTS
    return np.zeros(state_size)
