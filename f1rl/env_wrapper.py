import gym
import numpy as np
import yaml
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "true"
import pygame

MAP_SCALE_FACTOR = 0.8

class UnevenSignedActionRescale(gym.ActionWrapper):
    """
    Scales between two ranges containing zero, unevenly between negatives and positives.
    This maps zero to zero.
    """
    def __init__(self, env, low, high):
        super().__init__(env)
        low = np.broadcast_to(low, env.action_space.shape).astype(np.float32)
        high = np.broadcast_to(high, env.action_space.shape).astype(np.float32)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.pos_scale = env.action_space.high / self.action_space.high
        self.neg_scale = env.action_space.low / self.action_space.low

    def action(self, act):
        pos_mask = act >= 0
        act[pos_mask] *= self.pos_scale[pos_mask]
        act[~pos_mask] *= self.neg_scale[~pos_mask]
        return act

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
        self.screen = None
        self.font = None
        self.map_img = None
        self.map_cfg = None
        self.centerline = np.genfromtxt(centerline_path, delimiter=',', dtype=np.float32)

        self.action_space = gym.spaces.Box(np.array([steer_min, vel_min]), np.array([steer_max, vel_max]), dtype=np.float32)
        obs_shape = self._transform_state(env.reset(init_state_supplier(self))[0]).shape
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)

    def _transform_state(self, *args, **kwargs):
        return self.state_featurizer(self, *args, **kwargs)

    def _augment_curr_state(self, curr_state):
        """augments state dict in-place and returns it"""
        slip_angles = np.empty(self.env.num_agents)
        steer_angles = np.empty(self.env.num_agents)
        for i in range(self.env.num_agents):
            agent_state = self.env.sim.agents[i].state
            slip_angles[i] = agent_state[6]
            signed_speed = agent_state[3]
            curr_state["linear_vels_x"][i] = signed_speed * np.cos(slip_angles[i])
            curr_state["linear_vels_y"][i] = signed_speed * np.sin(slip_angles[i])
            steer_angles[i] = agent_state[2]
        curr_state["slip_angles"] = slip_angles
        curr_state["steer_angles"] = steer_angles
        return curr_state

    def step(self, action):
        action = np.expand_dims(action, axis=0)
        for _ in range(self.action_repeat):
            next_state, _, done, info = self.env.step(action)
            if done:
                break
        features = self._transform_state(next_state, prev_state=self.curr_state, prev_action=action)
        reward = self.reward_fn(self.curr_state, action, next_state)
        self.curr_state = self._augment_curr_state(next_state)
        return features, reward, done, info
    
    def reset(self):
        init_state = self.init_state_supplier(self)
        next_state, *_ = self.env.reset(init_state)
        self.curr_state = self._augment_curr_state(next_state)
        return self._transform_state(self.curr_state)

    def _pos_m_to_px(self, screen, pose):
        origin = np.array(self.map_cfg["origin"][:2])
        m_per_px = self.map_cfg["resolution"] / MAP_SCALE_FACTOR
        point = pose[:-1]
        pos_px = np.around((point - origin) / m_per_px).astype(int)
        pos_px[1] = screen.get_height() - pos_px[1]
        return pos_px

    def _draw_car(self, screen, pose, color):
        origin = np.array(self.map_cfg["origin"][:2])
        m_per_px = self.map_cfg["resolution"] / MAP_SCALE_FACTOR
        point = pose[:-1]
        pos_px = np.around((point - origin) / m_per_px).astype(int)
        pos_px[1] = screen.get_height() - pos_px[1]
        car_len_px = self.env.params["length"] / m_per_px
        car_width_px = self.env.params["width"] / m_per_px
        theta = pose[-1]
        rotmat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        cob = np.array([[1., 0.], [0., -1.]]) # change of basis required to invert y axis
        car_poly = [
            pos_px + cob @ rotmat @ np.array([car_len_px/2, 0]),
            pos_px + cob @ rotmat @ np.array([-car_len_px/2, -car_width_px/2]),
            pos_px + cob @ rotmat @ np.array([-car_len_px/2, car_width_px/2])
        ]
        car_poly = [np.around(x).astype(int) for x in car_poly]
        pygame.draw.polygon(screen, color, car_poly)

    def render(self, mode="human"):
        if mode != "human" and mode != "human_zoom":
            return super().render(mode)
        pygame.init()
        if self.font is None:
            self.font = pygame.font.SysFont(None, 24)
        if self.map_img is None:
            self.map_img = pygame.image.load(self.env.map_name + self.env.map_ext)
            self.map_img = pygame.transform.rotozoom(self.map_img, 0, MAP_SCALE_FACTOR)
        if self.map_cfg is None:
            with open(self.env.map_name + ".yaml", "r") as f:
                self.map_cfg = yaml.load(f, Loader=yaml.FullLoader)
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.map_img.get_width(), self.map_img.get_height() * 3 // 5))
        car_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255)
        ]
        map_screen = pygame.Surface(self.map_img.get_size())
        map_screen.blit(self.map_img, (0, 0))
        for i in range(self.env.num_agents):
            pose = np.array([self.curr_state[s][i] for s in ["poses_x", "poses_y", "poses_theta"]])
            self._draw_car(map_screen, pose, car_colors[i])
        if mode == "human":
            self.screen.blit(map_screen, (0, 0), area=pygame.Rect(0, map_screen.get_height() / 5, map_screen.get_width(), 3 * map_screen.get_height() / 5))
        else:
            def blitRotate(surf, image, pos, originPos, angle):
                image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
                offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
                rotated_offset = offset_center_to_pivot.rotate(-angle)
                rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)
                rotated_image = pygame.transform.rotate(image, angle)
                rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)
                surf.blit(rotated_image, rotated_image_rect)
            pose = np.array([self.curr_state[s][0] for s in ["poses_x", "poses_y", "poses_theta"]])
            car_pos_px = pygame.Vector2(*self._pos_m_to_px(map_screen, pose))
            intermediate = pygame.Surface([x//2 for x in self.screen.get_size()])
            inter_center = pygame.Vector2(*self.screen.get_size())/4
            blitRotate(intermediate, map_screen, inter_center, car_pos_px, 90 - pose[2]*180/np.pi)
            intermediate = pygame.transform.rotozoom(intermediate, 0, 2)
            self.screen.blit(intermediate, (0,0))
        img = self.font.render(f"{self.curr_state['linear_vels_x'][0]:.2f} m/s", True, (0, 0, 255))
        self.screen.blit(img, (20, 20))
        pygame.display.flip()
