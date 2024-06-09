from popgym.envs.lunar_lander_continuous_mask_velocities_multi_discrete import LunarLanderContinuousMaskVelocitiesMultiDiscrete

if __name__ == "__main__":
    game = LunarLanderContinuousMaskVelocitiesMultiDiscrete(render_mode="human")
    done = False
    obs, info = game.reset()
    reward = -float("inf")
    game.render()

    while not done:
        action = game.action_space.sample()
        obs, reward, done, truncated, info = game.step(action)
        game.render()
        print("reward:", reward)
