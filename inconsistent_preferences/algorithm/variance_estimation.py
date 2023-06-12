from typing import Optional, Sequence, Tuple, cast

import torch as th
from imitation.algorithms import preference_comparisons
from imitation.data.types import Transitions


class MeanAndVariancePreferenceModel(preference_comparisons.PreferenceModel):
    def rewards(self, transitions: Transitions) -> th.Tensor:
        preprocessed = self.model.preprocess(
            state=transitions.obs,
            action=transitions.acts,
            next_state=transitions.next_obs,
            done=transitions.dones,
        )
        rews = self.model(*preprocessed)
        assert rews.shape == (len(transitions), 2)
        return rews

    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        means1 = rews1[:, 0]
        means2 = rews2[:, 0]
        log_stds1 = rews1[:, 1]
        log_stds2 = rews2[:, 1]
        stds1 = log_stds1.exp()
        stds2 = log_stds2.exp()

        if self.discount_factor == 1:
            diff_mean = (means2 - means1).sum(axis=0)
            diff_std = (stds1**2 + stds2**2).sum(axis=0) ** 0.5
        else:
            discounts = self.discount_factor ** th.arange(len(rews1))
            diff_mean = (discounts * (means2 - means1)).sum(axis=0)
            diff_std = ((discounts * stds1) ** 2 + (discounts * stds2) ** 2).sum(
                axis=0
            ) ** 0.5

        diff_z_score = diff_mean / diff_std
        model_probability = None

        # Clip to avoid overflows (which in particular may occur
        # in the backwards pass even if they do not in the forward pass).
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        model_probability = 1 / (1 + returns_diff.exp())
        probability = self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability
        if self.ensemble_model is not None:
            assert probability.shape == (self.model.num_members,)
        else:
            assert probability.shape == ()
        return probability
