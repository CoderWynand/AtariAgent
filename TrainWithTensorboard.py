import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=None)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env

def chooseModel():
    env = DummyVecEnv([make_env])
    log_path = "./logs/dqn_spaceinvaders_tensorboard/"
    new_logger = configure(log_path, ["stdout", "tensorboard"])

    while (True):
        print("Select 1 to create a new agent, or 2 to load an existing agent")

        try:
            number = int(input("Enter an integer: "))
            print("You entered:", number)

            if(number ==1 or number ==2):
                break
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("That wasn't a valid integer.")

    if (number ==1):
        model = DQN("CnnPolicy", env, verbose=1, buffer_size=100_000, learning_rate=1e-4,
            learning_starts=10_000, target_update_interval=1000, train_freq=4, gamma=0.99, tensorboard_log="./logs/dqn_spaceinvaders_tensorboard/")
    else:
        path =  input("Please enter the file path of the existing agent")
        model = DQN.load(path, env=env, tensorboard_log=log_path)
        

    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=50_000,
                                            save_path = "./checkpoints",
                                            name_prefix = "dqn_spaceinvader")

    
    model.learn(total_timesteps=1000, tb_log_name = "DQN_SpaceInvaders", callback=checkpoint_callback)

    model.save("dqn_SpaceInvaders")
    

    env.close()

chooseModel()







