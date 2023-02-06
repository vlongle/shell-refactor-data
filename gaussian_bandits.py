"""
Reference: https://github.com/bgalbraith/bandits
"""
import numpy as np


class NormalDistribution:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self) -> str:
        return f"N({self.mu:.3f}, {self.sigma:.3f})"

    def __str__(self) -> str:
        return self.__repr__()

    def pdf(self, x: float) -> float:
        """
        Returns the probability density function for the bandit
        """
        return np.exp(-(x - self.mu)**2 / (2 * self.sigma**2)) / np.sqrt(2 * np.pi * self.sigma**2)

    def sample(self) -> float:
        """
        Returns a sample from the bandit
        """
        return np.random.normal(self.mu, self.sigma)


class GaussianBanditArm:
    def __init__(self, mu: float, sigma: float):
        self.dist = NormalDistribution(mu, sigma)

    def pull(self) -> float:
        """
        Returns the reward for the bandit
        """
        return self.dist.sample()

    def __repr__(self) -> str:
        return f"{self.dist}"

    def __str__(self) -> str:
        return self.__repr__()


# Thompson Sampling for Gaussian Bandits
"""
https://github.com/WhatIThinkAbout/BabyRobot/blob/master/Multi_Armed_Bandits/Part%205b%20-%20Thompson%20Sampling%20using%20Conjugate%20Priors.ipynb
"""


class ThompsonSampling:
    """
    Unknown mean, unknown variance, Gaussian bandits
    Use NormalGamma distribution as the conjugate prior
    (http://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-4/04-normal-gamma.pdf)
    """

    def __init__(self, bandits: list):
        self.bandits = bandits
        self.n_bandits = len(bandits)
        # assume that precision tau ~ Gamma(alpha, beta)
        # assume that mean mu | tau ~ Normal(c, (c * tau)**-1)

        # prior parameters: m, c, alpha, beta
        self.alpha = np.ones(self.n_bandits)
        self.beta = np.ones(self.n_bandits)

        # self.m = np.zeros(self.n_bandits)

        # optimistic
        self.m = np.ones(self.n_bandits) * 10
        self.c = np.ones(self.n_bandits)

        # store regret
        self.regrets = []

    def update(self, arm: int, reward: float):
        """
        Update the posterior parameters
        """
        self.alpha[arm] += 0.5
        self.beta[arm] += 0.5 * (reward - self.m[arm])**2

        self.m[arm] = (self.c[arm] * self.m[arm] + reward) / (self.c[arm] + 1)
        self.c[arm] += 1

        # update regret
        self.regrets.append(self.best_arm_reward - reward)

    @property
    def best_arm_reward(self) -> float:
        """
        Returns the reward of the best arm
        """
        return max([bandit.pull() for bandit in self.bandits])

    def select_arm(self) -> int:
        """
        Select the arm with the highest expected reward
        """
        # sample from the posterior distribution
        tau = np.random.gamma(self.alpha, 1 / self.beta)
        mu = np.random.normal(self.m, np.sqrt(1 / (self.c * tau)))
        return np.argmax(mu)
        # samples = np.random.normal(mu, np.sqrt(1 / tau))
        # return np.argmax(samples)

    @property
    def posterior(self):
        tau = np.random.gamma(self.alpha, 1 / self.beta)
        mu = np.random.normal(self.m, np.sqrt(1 / (self.c * tau)))
        return [NormalDistribution(mu[i], np.sqrt(1 / tau[i])) for i in range(self.n_bandits)]

    @property
    def cumulative_regret(self) -> list:
        return np.cumsum(self.regrets)


class LifelongThompsonSampling(ThompsonSampling):
    def __init__(self, bandits: list, num_cls_per_task: int):
        super().__init__(bandits)
        self.num_cls_per_task = num_cls_per_task

    def get_available_arms(self, ll_time: int) -> list:
        return [i for i in range((ll_time + 1) * self.num_cls_per_task)]

    def select_arm(self, ll_time: int) -> int:
        """
        Select the arm with the highest expected reward
        """
        # sample from the posterior distribution
        ll_arms = self.get_available_arms(ll_time)
        tau = np.random.gamma(self.alpha[ll_arms], 1 / self.beta[ll_arms])
        mu = np.random.normal(self.m[ll_arms], np.sqrt(
            1 / (self.c[ll_arms] * tau)))
        return np.argmax(mu)

    def learn_tasks(self, ll_time: int, n_trials: int):
        rewards = []
        actions = []
        for i in range(n_trials):
            arm = self.select_arm(ll_time)
            actions.append(arm)
            reward = self.bandits[arm].pull()
            self.update(arm, reward)
            rewards.append(reward)
        return rewards, actions
