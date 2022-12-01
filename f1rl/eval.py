import argparse
import numpy as np
import os
import time
import yaml
import wandb
from datetime import datetime
import shutil

import d3rlpy
from d3rlpy.algos.sac import SAC

from .util import create_env_from_config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path, in the form entity/project/run")
    parser.add_argument("weights_file_basename", help="Basename of weights file to load, i.e. model_1000.pt")
    parser.add_argument("--gpu", action="store_true", help="Use gpu")
    parser.add_argument("-n", "--n_rollouts", type=int, default=10, help="Number of rollouts. (default 10)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--render", action="store_true", help="Render rollout")
    group.add_argument("-v", "--video", action="store_true", help="Save video")
    return parser.parse_args()

def main():
    args = get_args()
    assert not args.video, "Not supported yet!"

    tmp_dir = f"eval/eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(tmp_dir)

    def restore_file(path):
        return wandb.restore(path, run_path=args.run_path, root=tmp_dir)

    try:
        config_file_stream = restore_file("config.yml")
        config = yaml.load(config_file_stream, Loader=yaml.FullLoader)
        config["env"]["state_featurizer_path"] = restore_file(os.path.basename(config["env"]["state_featurizer_path"])).name
        config["env"]["reward_fn_path"] = restore_file(os.path.basename(config["env"]["reward_fn_path"])).name
        config["env"]["state_sampler_path"] = restore_file(os.path.basename(config["env"]["state_sampler_path"])).name

        params_file = restore_file("model/params.json").name
        weights_file = restore_file(f"model/{args.weights_file_basename}").name
        agent: d3rlpy.algos.AlgoBase = SAC.from_json(params_file, use_gpu=args.gpu)
        agent.load_model(weights_file)

        env = create_env_from_config(config)

        ep_returns = []
        ep_steps = []
        for _ in range(args.n_rollouts):
            state = env.reset()
            done = False
            ep_reward = 0
            step = 0
            while not done:
                action = agent.predict(state.reshape(1, -1)).flatten()
                state, reward, done, _ = env.step(action)
                ep_reward += reward
                step += 1
                if args.render:
                    env.render(mode="human")
                    time.sleep(env.timestep * config["env"]["action_repeat"])
            ep_returns.append(ep_reward)
            ep_steps.append(step)
        print(f"Average return: {np.mean(ep_returns)}, Average ep len: {np.mean(ep_steps)}")
    finally:
        print("Cleaning up and exiting...")
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
