import gymnasium
from neorl import core


def make_env(task):
    env = gymnasium.make(task)
    env_data = core.EnvData()
    env.set_name = env_data.set_name
    env.set_reward_func = env_data.set_reward_func
    env.get_reward_func = env_data.get_reward_func
    return env
