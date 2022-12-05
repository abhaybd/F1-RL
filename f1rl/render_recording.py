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

def main():
    args = get_args()
    with open(args.recording_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    states = data["states"] # [episode, agent, step] => (x, y, theta)
    episode = np.array(states[args.index][0])
    map = data["map"] if "map" in data else "maps/austin"
    img = Image.open(f"{map}.png")

    ps = episode[:,:2]
    dists = np.linalg.norm(ps[1:] - ps[:-1], axis=-1)
    print("Total dist:", np.sum(dists))

    resolution = 0.08089
    origin = np.array([-21.25772567260448,-70.80398789934522])
    points = (episode[:,:2] - origin) / resolution
    points[:,1] = img.height - points[:, 1]
    points = points[:len(points)//2]

    raceline = np.genfromtxt(f"{map}_raceline.csv", delimiter=";", dtype=np.float32)[:,1:3]
    raceline = (raceline - origin) / resolution
    raceline[:,1] = img.height - raceline[:,1]

    plt.imshow(img, "gray")
    plt.plot(points[:, 0], points[:, 1], color="red")
    plt.plot(raceline[:,0], raceline[:,1], "b--")
    plt.show()



if __name__ == "__main__":
    main()