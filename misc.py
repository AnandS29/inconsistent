import numpy as np

def collect_trajectories(env, policy, num_episodes, render=False):
    trajectories = []
    for _ in range(num_episodes):
        trajectory = []
        obs = env.reset()
        done = False
        while not np.any(done):
            action, _ = policy.predict(obs)
            next_obs, reward, done, info = env.step(action)
            trajectory.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            if render:
                env.render()
        trajectories.append(trajectory)
    return trajectories