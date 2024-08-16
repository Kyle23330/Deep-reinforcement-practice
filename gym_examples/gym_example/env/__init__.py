from gym.envs.registration import register

register(
    id='gym_example/CliffWorldEnv-v0',
    entry_point='gym_example.env.cliff_world_env:CliffWorldEnv',
    max_episode_steps=300,
)
register(
    id='gym_example/GridWorldEnv-v0',
    entry_point='gym_example.env.grid_world_env:GridWorldEnv',
    max_episode_steps=300,
)
