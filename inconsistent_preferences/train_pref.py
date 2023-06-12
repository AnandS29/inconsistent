import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from imitation.algorithms import preference_comparisons
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from .algorithm.gatherer import NoisyGatherer
from .envs.make_envs import make_env
from .misc import collect_trajectories


def train_pref(
    env_name: str,
    timesteps: int,
    epochs_reward: int,
    epochs_agent: int,
    comparisons: int,
    algo: str,
    seed: Optional[int],
    render: bool,
    eval: bool,
    verbose: bool,
    stats: bool,
    eval_episodes: int,
    iterations: int,
    parallel: int,
    initial_comparison_frac: float,
    noise: float,
):
    rng = np.random.default_rng(seed)

    ########## 2. Set up environment and algorithm ##########

    venv, noise_fn, frag_length = make_env(
        env_name, rng=rng, noise=noise, parallel=parallel
    )

    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )

    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, rng=rng)
    gatherer = NoisyGatherer(noise_fn=noise_fn, seed=0)
    preference_model = preference_comparisons.PreferenceModel(reward_net)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=epochs_reward,
        rng=rng,
    )

    make_agent = lambda: PPO(
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=venv,
        seed=0,
        n_steps=2048 // venv.num_envs,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.001,
        n_epochs=epochs_agent,
    )

    agent = make_agent()

    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0,
        rng=rng,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=iterations,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=frag_length,
        transition_oversampling=1,
        initial_comparison_frac=initial_comparison_frac,
        allow_variable_horizon=False,
        initial_epoch_multiplier=1,
    )

    ########## 3. Train reward function ##########

    pref_comparisons.train(
        total_timesteps=timesteps,
        total_comparisons=comparisons,
    )

    ########## 4. Train policy on learned reward ##########
    learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict)
    agent = make_agent()
    agent.learn(timesteps)

    ########## 5. Evaluate ##########
    if eval:
        reward, _ = evaluate_policy(agent.policy, venv, eval_episodes, render=render)

        print(f"Reward averaged over {eval_episodes} episodes: {reward}")

    ########## 6. Print stats ##########
    if stats:
        print("Saving stats...")
        filename = env_name + "/" + filename + "/"
        try:
            print("Making directory...")
            os.makedirs("plots/" + filename)
        except:
            print("Directory exists")

        print(f"File Dir: {filename}")

        if env_name == "linear1d":
            vals = []
            actions = np.arange(0, 1, 0.01)
            plt.title("Reward Function")
            for i in actions:
                val = reward_net.predict(
                    np.array([[0]]),
                    np.array([[i]]),
                    np.array([[0]]),
                    np.array([[True]]),
                )
                vals.append(val)
            plt.figure()
            plt.plot(actions, vals)
            plt.xlabel("Action")
            plt.ylabel("Reward")
            plt.savefig(f"plots/{filename}r_fn.png")
        if env_name == "multi1d":
            # Plot actions
            trajs = collect_trajectories(venv, learner.policy, eval_episodes)
            acts = {g: [] for g in goal}
            for traj in trajs:
                for t in range(len(traj)):
                    a = traj[t][1][0]
                    ind = 0
                    for g in goal:
                        acts[g].append(a[ind])
                        ind += 1
            plt.figure()
            plt.title("Action distribution")
            for g in goal:
                plt.hist(acts[g], 20, alpha=0.5, label=str(g))
                # print(acts[g])
            plt.legend()
            plt.savefig(f"plots/{filename}act_dist.png")
        if env_name == "linear2d":
            plt.figure()
            plt.title("Reward Function")
            xs = np.arange(0, 1, 0.01)
            ys = np.arange(0, 1, 0.01)
            f = lambda x, y: reward_net.predict(
                np.array([[0]]), np.array([[x, y]]), np.array([[0]]), np.array([[True]])
            )
            z = np.array([[f(x, y) for x in xs] for y in ys])
            # print(z.shape)
            plt.imshow(
                z.reshape(z.shape[:2]), extent=[0, 1, 0, 1], cmap=cm.jet, origin="lower"
            )
            plt.colorbar()
            plt.savefig(f"plots/{filename}r_fn.png")

            def find_optimal(fn, ub):
                argmax = None
                max_val = None
                for x in np.arange(0, ub, 0.1):
                    for y in np.arange(0, ub, 0.1):
                        val = fn(x, y)
                        if max_val is None or val >= max_val:
                            argmax = [x, y]
                            max_val = val
                return argmax

            plt.figure()
            plt.title("Optimal Values")
            ubs = list(np.arange(0.1, 10, 0.1))
            vals = [find_optimal(f, ub) for ub in ubs]
            plt.plot(ubs, [v[0] for v in vals], label="x")
            plt.plot(ubs, [v[1] for v in vals], label="y")
            plt.plot(ubs, ubs, label="True opt")
            plt.legend()
            plt.savefig(f"plots/{filename}opt_val.png")


if __name__ == "__main__":
    ########## 1. Parse arguments ##########
    # Example: python3 test_pref.py --env linear1d --stats --verbose

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear1d")
    parser.add_argument("--timesteps", type=int, default=2 * (10**7))
    parser.add_argument("--epochs_reward", type=int, default=3)
    parser.add_argument("--epochs_agent", type=int, default=10)
    parser.add_argument("--comparisons", type=int, default=300**2)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--initial_comparison_frac", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.0)

    args = parser.parse_args()

    train_pref(
        args.env,
        args.timesteps,
        args.epochs_reward,
        args.epochs_agent,
        args.comparisons,
        args.algo,
        args.seed,
        args.render,
        args.eval,
        args.verbose,
        args.stats,
        args.eval_episodes,
        args.iterations,
        args.parallel,
        args.initial_comparison_frac,
        args.noise,
    )
