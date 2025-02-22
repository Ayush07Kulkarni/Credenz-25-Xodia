import pygame
import numpy as np
import gym
from stable_baselines3 import SAC
import environment

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 450
FPS = 60
SLINGSHOT_POS = (100, 270)

# Load images
try:
    background = pygame.image.load("ui/images/bg.jpg")
    bird_img = pygame.image.load("ui/images/bird.png")
    pig_img = pygame.image.load("ui/images/pig.png")
    slingshot_img = pygame.image.load("ui/images/sling.png")
except pygame.error as e:
    print(f"Error loading images: {e}")
    exit()

# Resize images if needed
bird_img = pygame.transform.scale(bird_img, (40, 40))
pig_img = pygame.transform.scale(pig_img, (40, 40))
slingshot_img = pygame.transform.scale(slingshot_img, (80, 150))

# Initialize Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Angry Birds AI")

# Load the trained SAC model
model = SAC.load("sac_angry_birds_model.zip")

# Create environment
env = gym.make("AngryBirdsEnv-v0")  # Ensure this matches your registered environment
obs, _ = env.reset()

clock = pygame.time.Clock()
running = True
done = False
paused = False

while running:
    screen.blit(background, (0, 0))  # Draw background
    screen.blit(slingshot_img, (SLINGSHOT_POS[0] - 20, SLINGSHOT_POS[1] - 80))  # Slingshot position

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Reset on pressing 'R'
                obs, _ = env.reset()
            elif event.key == pygame.K_p:  # Pause on pressing 'P'
                paused = not paused

    if not paused:
        # Let the trained model decide the action
        action, _states = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, reward, done, _, _ = env.step(action)
        print(obs)
        # Get bird and pig positions
        bird_x, bird_y = int(obs[0]) , int(obs[1])
        pig_x, pig_y = int(obs[4]), int(obs[5])

        # Draw pig and bird
        screen.blit(pig_img, (pig_x , pig_y ))
        screen.blit(bird_img, (bird_x , bird_y))

    pygame.display.flip()
    clock.tick(FPS)

    # Reset if episode ends
    if done:
        obs, _ = env.reset()
        pygame.time.delay(1000)  # Pause for 1 second before restarting

pygame.quit()