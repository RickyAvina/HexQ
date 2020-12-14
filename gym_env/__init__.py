from gym.envs.registration import register


register(
    id='GridEnv-v0',
    entry_point='gym_env.grid_env.grid_env:GridEnv',
    kwargs={},
    max_episode_steps=1000
    )
