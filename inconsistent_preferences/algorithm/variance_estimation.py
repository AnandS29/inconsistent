from typing import Any, Dict

import gym
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.data.types import Transitions
from imitation.rewards import reward_nets
from imitation.util import networks
from stable_baselines3.common import preprocessing
from torch import distributions


class BaseRewardNetWithUncertainty(reward_nets.RewardNet):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        _out_size: int = 1,
        **kwargs,
    ):
        super().__init__(observation_space, action_space)
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        full_build_mlp_kwargs: Dict[str, Any] = {
            "hid_sizes": (32, 32),
            **kwargs,
            # we do not want the values below to be overridden
            "in_size": combined_size,
            "out_size": _out_size,
            "squeeze_output": False,
        }

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)

    def forward(self, state, action, next_state, done):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)

        return self.mlp(inputs_concat)

    def predict_th(self, state, action, next_state, done, **kwargs) -> th.Tensor:
        with networks.evaluating(self):
            state_th, action_th, next_state_th, done_th = self.preprocess(
                state,
                action,
                next_state,
                done,
            )
            with th.no_grad():
                rew_th: th.Tensor = self(
                    state_th, action_th, next_state_th, done_th, **kwargs
                )

            return rew_th

    def predict(self, state, action, next_state, done, **kwargs):
        rew_th = self.predict_th(state, action, next_state, done, **kwargs)
        return rew_th.detach().cpu().numpy().flatten()


class MeanAndVarianceRewardNet(BaseRewardNetWithUncertainty):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            use_done=use_done,
            _out_size=2,
            **kwargs,
        )

    def forward(self, state, action, next_state, done, *, return_vars=False):
        outputs = super().forward(state, action, next_state, done)
        batch_size = state.shape[0]
        assert outputs.shape == (batch_size, 2)

        if return_vars:
            return outputs
        else:
            return outputs[:, 0]


class MeanAndVariancePreferenceModel(preference_comparisons.PreferenceModel):
    def rewards(self, transitions: Transitions) -> th.Tensor:
        preprocessed = self.model.preprocess(
            state=transitions.obs,
            action=transitions.acts,
            next_state=transitions.next_obs,
            done=transitions.dones,
        )
        rews: th.Tensor = self.model(*preprocessed, return_vars=True)
        assert rews.shape == (len(transitions), 2)
        return rews

    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        if rews1.ndim == 1 or rews2.ndim == 1:
            return super().probability(rews1, rews2)

        means1 = rews1[:, 0]
        means2 = rews2[:, 0]
        log_stds1 = rews1[:, 1]
        log_stds2 = rews2[:, 1]
        stds1 = log_stds1.exp()
        stds2 = log_stds2.exp()

        diff_mean: th.Tensor
        diff_std: th.Tensor
        if self.discount_factor == 1:
            diff_mean = (means2 - means1).sum(dim=0)
            diff_std = (stds1**2 + stds2**2).sum(dim=0) ** 0.5
        else:
            discounts = self.discount_factor ** th.arange(len(rews1))
            diff_mean = (discounts * (means2 - means1)).sum(dim=0)
            diff_std = ((discounts * stds1) ** 2 + (discounts * stds2) ** 2).sum(
                dim=0
            ) ** 0.5

        diff_z_score: th.Tensor = diff_mean / diff_std
        model_probability = 1 - distributions.Normal(0, 1).cdf(diff_z_score)
        probability: th.Tensor = (
            self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability
        )
        assert probability.shape == ()
        return probability
