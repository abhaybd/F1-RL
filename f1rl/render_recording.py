import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", help="Path to recording file")
    parser.add_argument("-i", "--index", default=0, type=int, help="Index of trajectory in recording.")
    return parser.parse_args()

def trim_to_single_lap(points: np.ndarray):
    start = points[0]
    dists = np.linalg.norm(points - start.reshape(1, -1), axis=1)
    thresh = 0.1
    for _ in range(10):
        close_to_start = dists <= thresh
        cross_line = ~close_to_start[:-1] & close_to_start[1:]
        if not np.any(cross_line):
            thresh *= 2
            print(f"Expanding loop end search to {thresh}")
            continue
        idx = np.where(cross_line)[0][0]
        loop = np.vstack([points[:idx+1], [start]])
        return loop
    raise AssertionError("Agent does not complete a full lap!")

def main():
    args = get_args()
    with open(args.recording_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    states = data["states"] # [episode, agent, step] => (x, y, theta)
    episode = np.array(states[args.index][0])
    map = data["map"] if "map" in data else "maps/austin"
    img = Image.open(f"{map}.png")

    with open(f"{map}.yaml", "r") as f:
        map_cfg = yaml.load(f, Loader=yaml.FullLoader)
    resolution = map_cfg["resolution"]
    origin = np.array(map_cfg["origin"][:2])
    points = episode[:, :2]
    points = trim_to_single_lap(points)
    print("Traveled dist:", np.sum(np.linalg.norm(points[1:] - points[:-1], axis=-1)))
    print(f"Lap time (assuming 20hz control): {len(points) / 20:.3f}sec")
    points = (points - origin) / resolution
    points[:,1] = img.height - points[:, 1]

    raceline = np.genfromtxt(f"{map}_raceline.csv", delimiter=";", dtype=np.float32)[:,1:3]
    print("Raceline dist:", np.sum(np.linalg.norm(raceline[1:] - raceline[:-1], axis=-1)))
    raceline = (raceline - origin) / resolution
    raceline[:,1] = img.height - raceline[:,1]

    plt.imshow(img, "gray")
    plt.plot(points[:, 0], points[:, 1], color="red")
    plt.plot(raceline[:,0], raceline[:,1], "b--")
    plt.show()



if __name__ == "__main__":
    main()