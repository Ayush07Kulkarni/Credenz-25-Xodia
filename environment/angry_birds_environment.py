import numpy as np
import gym
from gym import spaces
import pygame
from models.pig import Pig
from models.bird import Bird


# Constants
WIDTH, HEIGHT = 800, 450
FPS = 60
SLINGSHOT_POS = (100, 270)


class AngryBirdsEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.prev_distance = 0
        self.bird = Bird(*SLINGSHOT_POS)
        self.pig = Pig(600, 320)

        self.observation_space = spaces.Box(
        low=np.array([0, 0, -30, -30, 0, 0, 0, -np.pi], dtype=np.float32),
        high=np.array([WIDTH, HEIGHT, 30, 30, WIDTH, HEIGHT, HEIGHT, np.pi], dtype=np.float32),
        dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([1, 1], dtype=np.float32),
            high=np.array([10,10], dtype=np.float32),
            dtype=np.float32
        )

        self.clock = pygame.time.Clock()
        self.current_step = 0
        self.max_steps = 200
        self.min_distance = float('inf')
        self.trajectory_variety = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.bird.reset()
        self.pig.reset()
        self.trajectory_variety = []

        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        self.min_distance = np.sqrt(dx * dx + dy * dy)

        return self._get_obs(), {}

    def _get_obs(self):
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y

        return np.array([
            self.bird.x,
            self.bird.y,
            self.bird.velocity[0],
            self.bird.velocity[1],
            self.pig.x,
            self.pig.y,
            self.bird.max_height,
            self.bird.launch_angle
        ], dtype=np.float32)

    def get_reward_and_status(self):
        reward = 0
        done = False

        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        distance = np.sqrt(dx**2 + dy**2)

        
        improvement = self.prev_distance - distance
        reward += min(3 * improvement, 20)

       
        if distance < 40:  
            reward += 200
            print("Pig Killed!")
            done = True

        
        if self.bird.x > 850 or self.bird.y > 500 or self.bird.y < -50:
            reward -= 10  
            done = True
        
        
        if self.current_step >= self.max_steps:
            done = True
            reward -= 10

        
        if self.pig.x - self.bird.x < 0:
            reward -= 20

        
        if 0 < self.pig.x - self.bird.x < 200 and 0 < self.pig.y - self.bird.y < 100:
            reward += 10  

        
        power = 0.5 * np.sqrt(self.bird.velocity[0]**2 + self.bird.velocity[1]**2)
        reward -= 0.01 * power  

       
        reward -= 0.02 * self.current_step  

        slope = (self.pig.y - self.bird.y) / (self.pig.x - self.bird.x)
        min_angle = np.arctan(slope) - np.pi/10
        max_angle = np.arctan(slope) + np.pi/10
        if min_angle <= self.bird.launch_angle <= max_angle:
            reward += 70

        self.prev_distance = distance
        return reward, done



    def step(self, action):
        self.current_step += 1

        if not self.bird.launched:
            power_x = np.clip(action[0], 1, min(8, (self.pig.x - self.bird.x) / 30))
            power_y = np.clip(action[1], 1, min(8, (self.pig.y - self.bird.y) / 40))
            self.bird.launch(power_x, power_y)

        self.bird.update()

        
        dx = self.pig.x - self.bird.x
        dy = self.pig.y - self.bird.y
        current_distance = np.sqrt(dx * dx + dy * dy)
        self.min_distance = min(self.min_distance, current_distance)

        
        reward, done = self.get_reward_and_status()

        return self._get_obs(), reward, done, False, {}
