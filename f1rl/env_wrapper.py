import gym
import numpy as np

MAX_LASER_RANGE = 30

class F1EnvWrapper(gym.Wrapper):
    env: gym.Env
    def __init__(self, env, init_state_supplier, state_featurizer, reward_fn, action_repeat=1):
        super().__init__(env)
        steer_min = env.params["s_min"]
        steer_max = env.params["s_max"]
        vel_min = env.params["v_min"]
        vel_max = env.params["v_max"]

        self._init_state_supplier = init_state_supplier
        self.action_repeat = action_repeat
        self.reward_fn = reward_fn
        self.state_featurizer = state_featurizer
        self.curr_state = None

        self.action_space = gym.spaces.Box(np.array([steer_min, vel_min]), np.array([steer_max, vel_max]), dtype=np.float32)
        obs_shape = state_featurizer(env.reset()).shape
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)

    def step(self, action):
        for _ in range(self.action_repeat):
            next_state, _, done, info = self.env.step(action)
        features = self.state_featurizer(self, next_state)
        reward = self.reward_fn(self.curr_state, action, next_state)
        self.curr_state = next_state
        return features, reward, done, info
    
    def reset(self):
        init_state = self._init_state_supplier()
        next_state, *_ = self.env.reset(init_state)
        self.curr_state = next_state
        return self.state_featurizer(next_state)
