from f110_gym.envs.base_classes import Integrator
import yaml
import gym

from f1rl.env_wrapper import F1EnvWrapper, UnevenSignedActionRescale

def get_fn_from_file(path, fn_name):
    with open(path) as f:
        code = compile(f.read(), path, "exec")
    d = {}
    exec(code, d, d)
    return d[fn_name]

def create_env_from_config(config, seed=None, finite_horizon=True):
    env_kwargs = {
        "num_agents": 1,
        "map": config["env"]["map"]
    }
    if seed is not None:
        env_kwargs["seed"] = seed
    env = gym.make("f110_gym:f110-v0", integrator=Integrator.Euler, **env_kwargs)
    state_featurizer = get_fn_from_file(config["env"]["state_featurizer_path"], "transform_state")
    reward_fn = get_fn_from_file(config["env"]["reward_fn_path"], "get_reward")
    create_state_sampler = get_fn_from_file(config["env"]["state_sampler_path"], "create_state_sampler")
    init_state_sampler = create_state_sampler(seed)
    action_repeat = config["env"]["action_repeat"]
    env = F1EnvWrapper(env, init_state_sampler, state_featurizer, reward_fn, action_repeat=action_repeat)
    env = UnevenSignedActionRescale(env, -1, 1)
    if finite_horizon:
        env = gym.wrappers.TimeLimit(env, config["env"]["horizon"])
    return env

def load_env_from_config_path(config_path: str, seed=None):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return create_env_from_config(config, seed=seed)
