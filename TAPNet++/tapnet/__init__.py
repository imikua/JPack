from gymnasium.envs.registration import register
# from .envs import *

register(
    id='tapnet/TAP-v0',
    entry_point='tapnet.envs:TAP',
    max_episode_steps=5000,
)