# Make gym environment that wraps another environment and overrides the reward
# function.

from typing import Union

import gym
import numpy as np


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_fn, info_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.env = env
        self.info_fn = info_fn

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (
            obs,
            self.reward_fn(obs, action, reward, done, info),
            done,
            self.info_fn(obs, action, reward, done, info),
        )

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class ActiveReacherEnv(gym.Env):
    is_env_goal: Union[int, bool]

    def __init__(self, choose_prob, pref_goal, debug=False):
        self.env = gym.make("Reacher-v2")

        obs_space = self.env.observation_space
        low, high = obs_space.low, obs_space.high
        goal_low = [0, 0]
        goal_high = [2 * np.pi, 0.2]
        self.observation_space = obs_space

        act_space = self.env.action_space
        low, high = act_space.low, act_space.high
        low = np.concatenate([low, goal_low])
        high = np.concatenate([high, goal_high])
        self.action_space = gym.spaces.Box(low=np.array(low), high=np.array(high))

        self.goal = None

        self.choose_prob = choose_prob
        self.pref_goal = pref_goal

        self.is_goal_selected = False
        self.is_env_goal = True

        self.debug = debug

    def action_to_goal(self, action):
        action_g = action[len(action) - 2 :]
        theta, r = action_g[0], action_g[1]
        goal = np.array([r * np.cos(theta), r * np.sin(theta)])
        return goal

    def step(self, action):
        obs, reward, done, info = self.env.step(action[: len(action) - 2])
        self.chosen_goal = self.action_to_goal(action)  # Goal chosen by the agent
        env_goal = obs[4:6]  # Goal in the environment

        is_goal_sel_before = 1 if self.is_goal_selected else 0
        if not self.is_goal_selected:  # Only select goal once
            if np.random.rand() < self.choose_prob:
                self.goal = self.chosen_goal
                self.is_env_goal = 0
            else:
                self.goal = env_goal
                self.is_env_goal = 1
            # obs = self.transform_obs(obs)
            self.is_goal_selected = True

        obs = self.transform_obs(obs)  # Make observation relative to the goal
        pref_goal_bonus = -np.linalg.norm(self.goal - self.pref_goal)
        dist_to_goal = -np.linalg.norm(obs[8:10])
        action_reward = -np.sum(action[: len(action) - 2] ** 2)
        info = {
            "pref_goal_bonus": pref_goal_bonus,
            "dist_to_goal": dist_to_goal,
            "chosen_to_pref": -np.linalg.norm(self.chosen_goal - self.pref_goal),
            "action_reward": action_reward,
            "env_goal": env_goal,
            "chosen_goal": self.chosen_goal,
            "dist_env_to_chosen": -np.linalg.norm(env_goal - self.chosen_goal),
            "is_env_goal": self.is_env_goal,
            "is_goal_selected": is_goal_sel_before,
            "goal": self.goal,
        }
        r = pref_goal_bonus + dist_to_goal  # + action_reward
        if self.debug:
            r = pref_goal_bonus
        return obs, r, done, info

    def reset(self):
        obs = self.env.reset()
        self.is_goal_selected = False
        self.is_env_goal = True

        return obs

    def transform_obs(self, obs):
        obs[8:10] = obs[8:10] + obs[4:6] - self.goal
        obs[4:6] = self.goal
        return obs

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class ReacherRewardWrapper(gym.Wrapper):
    def __init__(self, aug):
        super().__init__(gym.make("Reacher-v2"))
        self.env = gym.make("Reacher-v2")
        self.augment = aug

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # pdb.set_trace()
        dist = np.linalg.norm(obs[8:10])
        reward = -(dist**2) - np.log(dist)  # - np.sum(action**2)
        reward, done, info = self.augment(obs, action, reward, done, info)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class HopperRewardWrapper(gym.Wrapper):
    def __init__(self):
        super().__init__(gym.make("Hopper-v2"))
        self.env = gym.make("Hopper-v2")

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class SwimmerRewardWrapper(gym.Wrapper):
    def __init__(self):
        super().__init__(gym.make("Swimmer-v2"))
        self.env = gym.make("Swimmer-v2")

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class InvertedDoublePendulumRewardWrapper(gym.Wrapper):
    def __init__(self):
        super().__init__(gym.make("InvertedDoublePendulum-v2"))
        self.env = gym.make("InvertedDoublePendulum-v2")

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class StatelessEnv(gym.Env):
    def __init__(self, action_dim, reward_fn, info_fn):
        super(StatelessEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(action_dim,))
        self.observation_space = gym.spaces.Box(
            low=0, high=0, shape=(1,), dtype=np.uint8
        )  # No obs
        self.reward_fn = reward_fn
        self.info_fn = info_fn

    def step(self, action):
        reward = self.reward_fn(action)
        reward = float(reward)
        observation = self.observation_space.sample()
        done = True
        info: dict = {}
        return (
            observation,
            reward,
            done,
            self.info_fn(observation, action, reward, done, info),
        )

    def reset(self):
        observation = self.observation_space.sample()
        return observation

    def render(self):
        raise NotImplementedError

    # def close (self):
    #     ...


def register_fb_env(r_fn, action_dim, info_fn=None):
    if info_fn is None:
        info_fn = lambda obs, action, reward, done, info: info
    gym.envs.register(
        id="StatelessEnv-v0",
        entry_point=StatelessEnv,
        max_episode_steps=150,
        kwargs={"reward_fn": r_fn, "action_dim": action_dim, "info_fn": info_fn},
    )


def register_reward_env(env, r_fn, info_fn=None, debug=False):
    if info_fn is None:
        info_fn = lambda obs, action, reward, done, info: info
    gym.envs.register(
        id="RewardWrapper-v0",
        entry_point=RewardWrapper,
        max_episode_steps=150,
        kwargs={
            "env": env,
            "reward_fn": r_fn,
            "debug": debug,
            "info_fn": info_fn,
        },
    )


def register_multi_base_env():
    gym.envs.register(
        id="HopperRewardWrapper-v0",
        entry_point="inconsistent.envs:HopperRewardWrapper",
        max_episode_steps=150,
    )

    gym.envs.register(
        id="SwimmerRewardWrapper-v0",
        entry_point="inconsistent.envs:SwimmerRewardWrapper",
        max_episode_steps=150,
    )

    gym.envs.register(
        id="InvertedDoublePendulumRewardWrapper-v0",
        entry_point="inconsistent.envs:InvertedDoublePendulumRewardWrapper",
        max_episode_steps=150,
    )
