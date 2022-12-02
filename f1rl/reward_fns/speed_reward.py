import numpy as np
COLLISION_WEIGHT = 10
CENTRIPETAL_WEIGHT = 0.1
SKID_WEIGHT = 1.0

def get_reward(state, action, next_state):
    idx = next_state["ego_idx"]
    vel_term = next_state["linear_vels_x"][idx]
    if next_state["collisions"][idx]:
        collision_term = COLLISION_WEIGHT * np.hypot(next_state["linear_vels_x"][idx], next_state["linear_vels_y"][idx])
    else:
        collision_term = 0
    centripetal_term = CENTRIPETAL_WEIGHT * vel_term ** 2 * np.abs(np.tan(action[0, 0]))
    skid_term = SKID_WEIGHT * np.abs(next_state["linear_vels_y"][idx])
    return vel_term - collision_term - centripetal_term - skid_term
