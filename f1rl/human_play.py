from f110_gym.envs.base_classes import Integrator
import pygame
import gym
import numpy as np
import time

from .env_wrapper import F1EnvWrapper, UnevenSignedActionRescale
from .state_samplers.constant_state_sampler import create_state_sampler

ACTION_REPEAT = 5

def get_action(joystick: pygame.joystick.Joystick):
    if joystick is not None:
        brake = joystick.get_axis(2)
        accel = joystick.get_axis(5)
        steer = joystick.get_axis(0)
        print(brake, accel, steer)
        return np.zeros(2)
    else:
        keys = pygame.key.get_pressed()
        steer, accel = 0.0, 0.0
        if keys[pygame.K_UP]:
            accel = 1.0
        elif keys[pygame.K_DOWN]:
            accel = -1.0
        if keys[pygame.K_RIGHT]:
            steer = -1.0
        elif keys[pygame.K_LEFT]:
            steer = 1.0
        return np.array([steer, accel])

def main():
    env = gym.make("f110_gym:f110-v0", map="maps/austin", num_agents=1, integrator=Integrator.Euler)
    sampler = create_state_sampler()
    env = F1EnvWrapper(env,sampler, lambda *_, **__: np.zeros(1), lambda *_, **__: 0, action_repeat=ACTION_REPEAT)
    env = UnevenSignedActionRescale(env, -1, 1)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        print("Using joystick!")
    else:
        print("No joystick found, using arrow keys!")
        joystick = None

    env.reset()
    done = False
    ep_reward = 0
    step = 0
    start_printing = False
    while not done:
        action = get_action(joystick)
        if start_printing or np.any(action != 0):
            start_printing = True
            print(env.sim.agents[0].state)
            print(action)
        _, reward, done, _ = env.step(action)
        ep_reward += reward
        step += 1
        env.render(mode="human")
        pygame.event.pump()
        time.sleep(env.timestep * ACTION_REPEAT)
    print(env.sim.agents[0].state)
    print(f"Accumulated {ep_reward} in {step} steps")

if __name__ == "__main__":
    main()