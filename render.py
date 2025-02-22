import pygame
import numpy as np
import gym
from stable_baselines3 import SAC
import environment


pygame.init()


WIDTH, HEIGHT = 800, 450
FPS = 60
SLINGSHOT_POS = (100, 270)


try:
    background = pygame.image.load("ui/images/bg.jpg")
    bird_img = pygame.image.load("ui/images/bird.png")
    pig_img = pygame.image.load("ui/images/pig.png")
    slingshot_img = pygame.image.load("ui/images/sling.png")
except pygame.error as e:
    print(f"Error loading images: {e}")
    exit()

bird_img = pygame.transform.scale(bird_img, (40, 40))
pig_img = pygame.transform.scale(pig_img, (40, 40))
slingshot_img = pygame.transform.scale(slingshot_img, (80, 150))


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Angry Birds AI")

model = SAC.load("sac_angry_birds_model.zip")


env = gym.make("AngryBirdsEnv-v0") 
obs, _ = env.reset()

clock = pygame.time.Clock()
running = True
done = False
paused = False

while running:
    screen.blit(background, (0, 0))
    screen.blit(slingshot_img, (SLINGSHOT_POS[0] - 20, SLINGSHOT_POS[1] - 80)) 

 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  
                obs, _ = env.reset()
            elif event.key == pygame.K_p:
                paused = not paused

    if not paused:
        
        action, _states = model.predict(obs, deterministic=True)

        
        obs, reward, done, _, _ = env.step(action)
        print(obs)
      
        bird_x, bird_y = int(obs[0]) , int(obs[1])
        pig_x, pig_y = int(obs[4]), int(obs[5])

       
        screen.blit(pig_img, (pig_x , pig_y ))
        screen.blit(bird_img, (bird_x , bird_y))

    pygame.display.flip()
    clock.tick(FPS)

    if done:
        obs, _ = env.reset()
        pygame.time.delay(1000)  

pygame.quit()
