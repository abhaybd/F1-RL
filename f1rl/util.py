import yaml
import gym
import numpy as np

from f1rl.env_wrapper import F1EnvWrapper

def get_fn_from_file(path, fn_name):
    with open(path) as f:
        code = compile(f.read(), path, "exec")
    d = {}
    exec(code, d, d)
    return d[fn_name]

def create_env_from_config(config, seed=None):
    if seed is None:
        env = gym.make("f110_gym:f110-v0", num_agents=1)
    else:
        env = gym.make("f110_gym:f110-v0", num_agents=1, seed=seed)
    state_featurizer = get_fn_from_file(config["env"]["state_featurizer_path"], "transform_state")
    reward_fn = get_fn_from_file(config["env"]["reward_fn_path"], "get_reward")
    create_state_sampler = get_fn_from_file(config["env"]["state_sampler_path"], "create_state_sampler")
    init_state_sampler = create_state_sampler(env, seed)
    action_repeat = config["env"]["action_repeat"]
    env = F1EnvWrapper(env, init_state_sampler, state_featurizer, reward_fn, action_repeat=action_repeat)
    env = gym.wrappers.RescaleAction(env, -1, 1)
    return env

def load_env_from_config_path(config_path: str, seed=None):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return create_env_from_config(config, seed=seed)
