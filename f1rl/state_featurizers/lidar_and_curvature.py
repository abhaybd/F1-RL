import numpy as np

from f1rl.env_wrapper import F1EnvWrapper

N_LIDAR = 20
N_CURVATURE_POINTS = 10
CURVATURE_LOOKAHEAD = 0.5


def downsample(arr, target_len):
    assert len(arr) >= target_len
    factor = len(arr) / target_len
    if factor.is_integer():
        return arr[::int(factor)]
    else:
        assert False, "Interpolated downsampling not supported. Implement it?"


def get_closest_idx(pose, centerline):
    points = centerline[:, :2]
    pos = pose[:-1].reshape(1, -1)
    dists = np.linalg.norm(points - pos, axis=1)
    closest_idx = np.argmin(dists)
    return closest_idx


def angle_to_centerline(pose: np.ndarray, centerline: np.ndarray):
    points = centerline[:, :2]
    closest_idx = get_closest_idx(pose, centerline)
    next_idx = (closest_idx + 1) % len(centerline)
    centerline_angle = np.arctan2(
        *(points[next_idx] - points[closest_idx])[::-1])
    raw_angle_diff = pose[2] - centerline_angle
    angle = np.arctan2(np.sin(raw_angle_diff), np.cos(raw_angle_diff))
    return angle


def cross(*args, **kwargs) -> np.ndarray:
    return np.cross(*args, **kwargs)


def menger_curvature_loop(points: np.ndarray):
    points = np.vstack([points[-1], points, points[0]])
    forward_vecs = points[2:] - points[1:-1]
    back_vecs = points[:-2] - points[1:-1]
    forward_vecs /= np.linalg.norm(forward_vecs, axis=1).reshape(-1, 1)
    back_vecs /= np.linalg.norm(back_vecs, axis=1).reshape(-1, 1)
    sins = cross(forward_vecs, back_vecs)
    back_to_fronts = points[2:] - points[:-2]
    curvatures = 2 * sins / np.linalg.norm(back_to_fronts, axis=1)
    return curvatures


def get_future_curvatures(pose: np.ndarray, centerline: np.ndarray, n_points, point_dist):
    idx = get_closest_idx(pose, centerline)

    points = centerline[:, :2]
    # rearrange to start at current point
    points = np.vstack([points[idx:], points[:idx]])
    dists = np.concatenate(
        [np.zeros(1), np.linalg.norm(points[:-1] - points[1:], axis=1)])
    cum_dists = np.cumsum(dists)

    sample_dists = np.linspace(0, point_dist * (n_points - 1), n_points)

    curvature_idxs = np.around(
        np.interp(sample_dists, cum_dists, np.arange(len(dists)))).astype(int)

    curvatures = menger_curvature_loop(points)
    return curvatures[curvature_idxs]


def transform_state(env: F1EnvWrapper, state, prev_state=None, prev_action=None):
    idx = state["ego_idx"]
    pose = np.array([state[s][idx]
                    for s in ["poses_x", "poses_y", "poses_theta"]])
    pose[2] = np.mod(pose[2], 2 * np.pi)
    vel = np.array([state[s][idx] for s in ["linear_vels_x", "linear_vels_y"]])
    if prev_state is not None:
        last_vel = np.array([prev_state[s][idx] for s in ["linear_vels_x", "linear_vels_y"]])
        accel = vel - last_vel
    else:
        accel = np.zeros_like(vel)
    angle_of_attack = np.array([angle_to_centerline(pose, env.centerline)])
    scans = downsample(state["scans"][idx], N_LIDAR)
    if prev_action is not None:
        prev_steer = np.array([prev_action[idx, 0]])
    else:
        prev_steer = np.array([0.])
    collision = np.array([state["collisions"][idx]])
    future_curvatures = get_future_curvatures(
        pose, env.centerline, N_CURVATURE_POINTS, CURVATURE_LOOKAHEAD)
    return np.concatenate([vel, accel, angle_of_attack, scans, prev_steer, collision, future_curvatures])


def main():
    centerline = np.genfromtxt(
        "maps/austin_centerline.csv", delimiter=",", dtype=np.float32)
    points = centerline[:, :2]
    curvatures = menger_curvature_loop(points)
    origin = np.array([-21.25772567260448, -70.80398789934522])

    import matplotlib.pyplot as plt
    from PIL import Image
    IMG_SIZE_PX = 2000
    M_PER_PX = 0.08089
    _, ax = plt.subplots()
    track = Image.open("maps/austin.png")
    ax.imshow(track, cmap="gray")

    def interpolate_color(value):
        scale_fac = 2
        if value < 0:
            scale_dir = np.array([0, -1, -1])
        else:
            scale_dir = np.array([-1, -1, 0])
        return np.clip(np.array([1, 1, 1]) + np.abs(value) * scale_fac * scale_dir, 0, 1)
    for point, curvature in zip(points, curvatures):
        px_x, px_y = (point - origin) / M_PER_PX
        color = interpolate_color(curvature)
        circle = plt.Circle((px_x, IMG_SIZE_PX - px_y), 5, color=color)
        ax.add_patch(circle)
    plt.show()


if __name__ == "__main__":
    main()
