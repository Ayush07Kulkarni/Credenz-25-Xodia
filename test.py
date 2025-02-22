import gym
import numpy as np
import torch
from stable_baselines3 import SAC
from environment.angry_birds_environment import AngryBirdsEnv

# Load the trained SAC model
model_path = "sac_angry_birds_model.zip"
model = SAC.load(model_path)

# Create the Angry Birds environment
env = AngryBirdsEnv()

# Reset the environment
obs, _ = env.reset()

# Run the environment with the trained model
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
print(reward)
# Check if the bird hit the pig
if np.sqrt((obs[4] - obs[0])**2 + (obs[5] - obs[1])**2) < 80:
    print("Hit!")
else:
    print("Miss!")