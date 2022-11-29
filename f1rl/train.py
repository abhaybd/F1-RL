import gym
import argparse
import os
import yaml
import torch
import numpy as np
import shutil
import pickle

from tqdm import tqdm

from d3rlpy.algos.sac import SAC
from d3rlpy.online.buffers import ReplayBuffer


import wandb

from f1rl.env_wrapper import F1EnvWrapper
from .util import create_env_from_config, get_fn_from_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config file")
    parser.add_argument("--dryrun", action="store_true", help="Disable logging, used for short test runs.")
    return parser.parse_args()


# def eval_rollout(env: TetrisEnv, agent: TetrisDQN, n_rollouts, horizon=None, render=False):
#     rets = []
#     lines_cleared = []
#     ep_lens = []
#     for _ in tqdm(range(n_rollouts), leave=False):
#         done = False
#         state = env.reset()
#         total_rew = 0
#         step = 0
#         while not done and (horizon is None or step < horizon):
#             action = agent.get_action(state)
#             state, reward, done, *_ = env.step(action)
#             total_rew += reward
#             step += 1
#             if render:
#                 env.render()
#         rets.append(total_rew)
#         lines_cleared.append(state.cleared)
#         ep_lens.append(step)
#     return {
#         "eval_rollout_return": np.mean(rets),
#         "eval_rollout_lines": np.mean(lines_cleared),
#         "eval_rollout_len": np.mean(ep_lens)
#     }


def save_file(out_dir: str, file: str):
    path = shutil.copy2(file, os.path.join(out_dir, os.path.basename(file)))
    return os.path.relpath(path)

def save_file(out_dir: str, file: str):
    path = shutil.copy2(file, os.path.join(out_dir, os.path.basename(file)))
    return os.path.relpath(path)

def main():
    args = get_args()

    if args.dryrun:
        os.environ["WANDB_MODE"] = "disabled"

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    logged_config = {k: v for k, v in config.items() if k not in
                     {"name", "policy", "notes"}}
    wandb.init(project=f"F1RL_{config['policy']}", entity="f1rl",
               name=config["name"], notes=config["notes"], config=logged_config)

    out_dir: str = wandb.run.dir

    env = create_env_from_config(config, seed=config["train"]["seed"])
    eval_env = create_env_from_config(config, seed=config["eval"]["seed"])

    # copy reward function and config into out dir and update config
    config["train"]["reward_fn_path"] = save_file(out_dir, config["train"]["reward_fn_path"])
    # copy featurization into out dir and update config
    config["env"]["state_featurizer_path"] = save_file(out_dir, config["env"]["state_featurizer_path"])

    with open(os.path.join(out_dir, "config.yml"), "w") as f:
        yaml.dump(config, f)

    # mark all files in the out dir for uploading
    wandb.save(os.path.join(out_dir, "*"))

    sac = SAC(**config["agent"])
    buffer = ReplayBuffer(maxlen=config["train"]["buffer_size"], env=env)
    steps_per_epoch = config["eval"]["interval"]
    sac.fit_online(env, buffer, eval_env=eval_env,
        n_steps_per_epoch=steps_per_epoch,
        n_steps=config["train"]["n_steps"],
        save_interval=config["train"]["save_interval"] // steps_per_epoch,
        experiment_name=config["name"],
        tensorboard_dir="runs",
        save_metrics=not args.dryrun)


if __name__ == "__main__":
    main()
