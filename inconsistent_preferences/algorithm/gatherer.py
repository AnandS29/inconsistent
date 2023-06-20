from typing import Optional, Sequence, Tuple

import numpy as np
from imitation.algorithms import preference_comparisons
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRewPair
from imitation.util import logger as imit_logger


class NoisyGatherer(preference_comparisons.PreferenceGatherer):
    """Computes noisy preferences using perturbed ground-truth environment rewards."""

    def __init__(
        self,
        noise_fn,
        discount_factor: float = 1,
        seed: Optional[int] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the synthetic preference gatherer.

        Args:
            discount_factor: discount factor that is used to compute
                how good a fragment is. Default is to use undiscounted
                sums of rewards (as in the DRLHP paper).
            seed: seed for the internal RNG (only used if temperature > 0 and sample)
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger=custom_logger)
        self.discount_factor = discount_factor
        self.rng = np.random.default_rng(seed=seed)
        self.noise_fn = noise_fn

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Computes probability fragment 1 is preferred over fragment 2."""
        returns1, returns2 = self._reward_sums(fragment_pairs)
        comparison = (returns1 > returns2).astype("float32")
        return comparison

    def _perturb_rews(self, f1):
        return self.noise_fn(f1.obs, f1.acts, f1.rews, f1.infos)

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        rews1, rews2 = zip(
            *[
                (
                    rollout.discounted_sum(
                        self._perturb_rews(f1), self.discount_factor
                    ),
                    rollout.discounted_sum(
                        self._perturb_rews(f2), self.discount_factor
                    ),
                )
                for f1, f2 in fragment_pairs
            ],
        )
        return np.array(rews1, dtype=np.float32), np.array(rews2, dtype=np.float32)
