import numpy as np
from imitation.util.util import make_vec_env

from .reward_env import register_fb_env


def make_env(env_name, rng, noise=0, parallel=1):
    venv = None
    if env_name == "linear1d":
        env_name = "StatelessEnv-v0"

        def r_fn(x):
            return x

        env_kwargs = {"action_dim": 1, "r_fn": r_fn}
        register_fb_env(**env_kwargs)

        def noise_fn(obs, acts, rews, infos):
            # Change to include new noisy reward structure
            val = acts[0, 0]
            noise = 0
            if val >= noise:
                if np.random.random() < 0.5:
                    noise = val
                noise = -val
            return rews + noise

        frag_length = 1
    elif env_name == "multi1d":
        env_name = "StatelessEnv-v0"
        act_dim = 3
        goal = np.array([0.2, 0.5, 0.8])

        def r_fn(x):
            return -np.linalg.norm(x - goal) ** 2

        env_kwargs = {"action_dim": act_dim, "r_fn": r_fn, "info_fn": None}
        register_fb_env(**env_kwargs)

        def noise_fn(obs, acts, rews, infos):
            # Change to include new noisy reward structure
            noise = 0
            for val in goal:
                noise += np.random.normal(0, noise * val)
            return rews + noise

        frag_length = 1
    elif env_name == "linear2d":
        env_name = "StatelessEnv-v0"

        def r_fn(x):
            return x[0] + x[1]

        def noise_fn(obs, acts, rews, infos):
            return rews + np.random.normal(0, noise * (acts[0, 0] ** 2))

        env_kwargs = {"action_dim": 2, "r_fn": r_fn}
        register_fb_env(**env_kwargs)
        frag_length = 1
    else:
        frag_length = 100
        noise = 0

    if noise == 0:
        noise_fn = lambda obs, acts, rews, infos: rews  # noqa: F811

    venv = make_vec_env(env_name, n_envs=parallel, rng=rng)

    return venv, noise_fn, frag_length
