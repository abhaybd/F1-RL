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

def get_args(start_time):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path, in the form entity/project/run")
    parser.add_argument("weights_file_basename", help="Basename of weights file to load, i.e. model_1000.pt")
    parser.add_argument("--gpu", action="store_true", help="Use gpu")
    parser.add_argument("-n", "--n_rollouts", type=int, default=10, help="Number of rollouts. (default 10)")
    parser.add_argument("-l", "--local", action="store_true", help="Evaluate a local checkpoint")
    parser.add_argument("-s", "--save", nargs="?", default=None, const=f"eval_saved/eval_{start_time}.yml",
                help="Save location of evaluation replay yaml. If specified without value, saves to default path.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--render", action="store_true", help="Render rollout")
    group.add_argument("-v", "--video", action="store_true", help="Save video")
    return parser.parse_args()

def main():
    start_time = datetime.now().strftime('%Y%m%d%H%M%S')
    args = get_args(start_time)
    assert not args.video, "Not supported yet!"
    assert not args.save or (args.save and not os.path.isfile(args.save)), "Specified eval save path already exists!"

    tmp_dir = f"eval/eval_{start_time}"
    os.makedirs(tmp_dir)

    def restore_file(path):
        return wandb.restore(path, run_path=args.run_path, root=tmp_dir)

    try:
        if args.local:
            with open(os.path.join(args.run_path, 'config.yml')) as config_file_stream:
                config = yaml.load(config_file_stream, Loader=yaml.FullLoader)

            params_file = os.path.join(args.run_path, "model/params.json")
            weights_file = os.path.join(args.run_path, f"model/{args.weights_file_basename}")
        else:
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
        states = []
        for _ in range(args.n_rollouts):
            state = env.reset()
            rollout_states = [[[env.curr_state[s][i].item() for s in ["poses_x", "poses_y", "poses_theta"]]] for i in range(env.env.num_agents)]
            done = False
            ep_reward = 0
            step = 0
            prev_ts = time.time()
            while not done:
                action = agent.predict(state.reshape(1, -1)).flatten()
                state, reward, done, _ = env.step(action)
                for i in range(env.env.num_agents):
                    rollout_states[i].append([env.curr_state[s][i].item() for s in ["poses_x", "poses_y", "poses_theta"]])
                ep_reward += reward
                step += 1
                if args.render:
                    env.render(mode="human")
                    ts = time.time()
                    desired_sleep = env.timestep * config["env"]["action_repeat"]
                    elapsed = ts - prev_ts
                    if desired_sleep > elapsed:
                        time.sleep(desired_sleep - elapsed)
                    prev_ts = ts
            ep_returns.append(ep_reward.item())
            ep_steps.append(step)
            states.append(rollout_states)
        print(f"Average return: {np.mean(ep_returns)}, Average ep len: {np.mean(ep_steps)}")

        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            results = {"map": config["env"]["map"], "run": args.run_path, "weights_file": args.weights_file_basename, "states": states, "returns": ep_returns, "num_steps": ep_steps}
            with open(args.save, "w") as f:
                yaml.dump(results, f)
    finally:
        print("Cleaning up and exiting...")
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
