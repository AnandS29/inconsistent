from inconsistent_preferences.train_pref import train_pref


def test_1d():
    train_pref(
        env_name="linear1d",
        timesteps=100,
        epochs_reward=1,
        epochs_agent=1,
        comparisons=100,
        algo="ppo",
        seed=0,
        render=False,
        eval=False,
        verbose=False,
        stats=True,
        eval_episodes=100,
        iterations=1,
        parallel=1,
        initial_comparison_frac=0.1,
        noise=0.0,
        reward_model="default",
    )

def test_1d_mean_and_variance():
    train_pref(
        env_name="linear1d",
        timesteps=100,
        epochs_reward=1,
        epochs_agent=1,
        comparisons=100,
        algo="ppo",
        seed=0,
        render=False,
        eval=False,
        verbose=False,
        stats=True,
        eval_episodes=100,
        iterations=1,
        parallel=1,
        initial_comparison_frac=0.1,
        noise=0.0,
        reward_model="mean_and_variance",
    )
