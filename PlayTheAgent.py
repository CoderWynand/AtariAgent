import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

def make_env():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
    env = AtariWrapper(env)
    return env

env = DummyVecEnv([make_env])

model  = DQN.load("dqn_SpaceInvaders", env = env)

obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic = True)
    obs, reward, done, info = env.step(action)
    if done[0]:
        obs = env.reset()
