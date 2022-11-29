import gym
import numpy as np
import os

MAX_LASER_RANGE = 30

class F1EnvWrapper(gym.Wrapper):
    env: gym.Env
    def __init__(self, env, init_state_supplier, state_featurizer, reward_fn, action_repeat=1):
        super().__init__(env)
        steer_min = env.params["s_min"]
        steer_max = env.params["s_max"]
        vel_min = env.params["v_min"]
        vel_max = env.params["v_max"]
        centerline_path = env.map_name + '_centerline.csv'

        self.init_state_supplier = init_state_supplier
        self.action_repeat = action_repeat
        self.reward_fn = reward_fn
        self.state_featurizer = state_featurizer
        self.curr_state = None
        self.centerline = np.genfromtxt(centerline_path, delimiter=',')[:, :2]

        self.action_space = gym.spaces.Box(np.array([steer_min, vel_min]), np.array([steer_max, vel_max]), dtype=np.float32)
        obs_shape = self._transform_state(env.reset(init_state_supplier(self))).shape
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)

    def _transform_state(self, state):
        return self.state_featurizer(self, state)

    def step(self, action):
        action = np.expand_dims(action, axis=0)
        for _ in range(self.action_repeat):
            next_state, _, done, info = self.env.step(action)
        features = self._transform_state(next_state)
        reward = self.reward_fn(self.curr_state, action, next_state)
        self.curr_state = next_state
        return features, reward, done, info
    
    def reset(self):
        init_state = self.init_state_supplier(self)
        next_state, *_ = self.env.reset(init_state)
        self.curr_state = next_state
        return self._transform_state(next_state)
