import argparse
import os
import yaml
import shutil
import wandb

from d3rlpy.algos.sac import SAC
from d3rlpy.online.buffers import ReplayBuffer

from .util import create_env_from_config
from .state_samplers.constant_state_sampler import create_state_sampler as create_constant_sampler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config file")
    parser.add_argument("--gpu", action="store_true", help="Whether to use gpu or not")
    parser.add_argument("--dryrun", action="store_true", help="Disable logging, used for short test runs.")
    return parser.parse_args()

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

    wandb.tensorboard.patch(root_logdir="runs")
    logged_config = {k: v for k, v in config.items() if k not in
                    {"name", "policy", "notes"}}
    wandb.init(project=f"F1RL_{config['policy']}", entity="f1rl", config=logged_config,
               name=config["name"], notes=config["notes"], sync_tensorboard=True)

    out_dir: str = wandb.run.dir

    env = create_env_from_config(config, seed=config["train"]["seed"])
    eval_env = create_env_from_config(config, seed=config["eval"]["seed"], override_state_sampler=create_constant_sampler())

    # copy relevant files
    config["env"]["reward_fn_path"] = save_file(out_dir, config["env"]["reward_fn_path"])
    config["env"]["state_featurizer_path"] = save_file(out_dir, config["env"]["state_featurizer_path"])
    config["env"]["state_sampler_path"] = save_file(out_dir, config["env"]["state_sampler_path"])

    with open(os.path.join(out_dir, "config.yml"), "w") as f:
        yaml.dump(config, f)

    # mark all files in the out dir for uploading
    wandb.save(os.path.join(out_dir, "*"))

    sac = SAC(use_gpu=args.gpu, **config["agent"])
    buffer = ReplayBuffer(maxlen=config["train"]["buffer_size"], env=env)
    steps_per_epoch = config["eval"]["interval"]
    sac.fit_online(env, buffer, eval_env=eval_env,
        n_steps_per_epoch=steps_per_epoch,
        n_steps=config["train"]["n_steps"],
        random_steps=config["train"]["n_random_steps"],
        save_interval=config["train"]["save_interval"] // steps_per_epoch,
        experiment_name="model",
        logdir=out_dir,
        tensorboard_dir="runs",
        save_metrics=not args.dryrun,
        with_timestamp=False)


if __name__ == "__main__":
    main()
