from stable_baselines3.common.env_util import make_vec_env
from environment.angry_birds_environment import AngryBirdsEnv
from stable_baselines3 import SAC
def train_model():

    env = make_vec_env(AngryBirdsEnv)
    model = SAC('MlpPolicy', env, verbose=1 ,batch_size=64, learning_rate= 3e-4, buffer_size=1000000, learning_starts=1000, train_freq=1, gradient_steps=1, ent_coef= 'auto', target_update_interval=1, gamma=0.999,)
    model.learn(total_timesteps=500000)

    model.save("sac_angry_birds_model")

    # YOUR TRAINING LOGIC GOES HERE.

    # SAVE THE MODEL. SHARE IT LATER WITH US BEFORE GIVEN DEADLINE.


if __name__ == "__main__":
    train_model()
