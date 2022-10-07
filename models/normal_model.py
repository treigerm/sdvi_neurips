from ctypes import addressof
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import math
import logging
import copy
import numpy as np
import seaborn as sns

sns.reset_orig()

from .abstract_model import AbstractModel


def marginal_likelihood(data, likelihood_std, prior_mean, prior_std):
    """Calculate the marginal likelihood of a branch. Assumes we observe only
    a single data point.

    Taken from Section 2.5 at https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf.
    """
    likelihood_var = math.pow(likelihood_std, 2)
    prior_var = math.pow(prior_std, 2)

    first_term = likelihood_std / (
        (math.sqrt(2 * math.pi) * likelihood_std)
        * math.sqrt(prior_var + likelihood_var)
    )
    second_term = math.exp(
        -(math.pow(data, 2) / (2 * likelihood_var))
        - (math.pow(prior_mean, 2) / (2 * prior_var))
    )
    third_term = math.exp(
        (
            (prior_var * math.pow(data, 2) / likelihood_var)
            + (likelihood_var * math.pow(prior_mean, 2) / prior_var)
            + 2 * data * prior_mean
        )
        / (2 * (prior_var + likelihood_var))
    )
    return first_term * second_term * third_term


def posterior_params(data, likelihood_std, prior_mean, prior_std):
    """Calculate the posterior mean and standard deviation of a branch. Assumes we
    observe only a single data point."""
    prior_precision = 1 / math.pow(prior_std, 2)
    likelihood_precision = 1 / math.pow(likelihood_std, 2)
    post_mean = (prior_precision * prior_mean + likelihood_precision * data) / (
        prior_precision + likelihood_precision
    )
    post_std = 1 / (prior_precision + likelihood_precision)
    return post_mean, post_std


class NormalModel(AbstractModel):
    likelihood_std = 1
    prior_std = 1
    observed_data = 2

    @staticmethod
    def get_k(u):
        if u <= -4:
            return 0
        elif -4 < u <= -3:
            return 1
        elif -3 < u <= -2:
            return 2
        elif -2 < u <= -1:
            return 3
        elif -1 < u <= 0:
            return 4
        elif 0 < u <= 1:
            return 5
        elif 1 < u <= 2:
            return 6
        elif 2 < u <= 3:
            return 7
        elif 3 < u <= 4:
            return 8
        elif 4 < u:
            return 9
        else:
            raise ValueError(f"Ups this should not happen. u={u}")

    @staticmethod
    def prior_branch_prob(u):
        if u <= -4:
            return dist.Normal(0, 5).cdf(torch.tensor(-4))
        elif -4 < u <= -3:
            return dist.Normal(0, 5).cdf(torch.tensor(-3)) - dist.Normal(0, 5).cdf(
                torch.tensor(-4)
            )
        elif -3 < u <= -2:
            return dist.Normal(0, 5).cdf(torch.tensor(-2)) - dist.Normal(0, 5).cdf(
                torch.tensor(-3)
            )
        elif -2 < u <= -1:
            return dist.Normal(0, 5).cdf(torch.tensor(-1)) - dist.Normal(0, 5).cdf(
                torch.tensor(-2)
            )
        elif -1 < u <= 0:
            return dist.Normal(0, 5).cdf(torch.tensor(0)) - dist.Normal(0, 5).cdf(
                torch.tensor(-1)
            )
        elif 0 < u <= 1:
            return dist.Normal(0, 5).cdf(torch.tensor(1)) - dist.Normal(0, 5).cdf(
                torch.tensor(0)
            )
        elif 1 < u <= 2:
            return dist.Normal(0, 5).cdf(torch.tensor(2)) - dist.Normal(0, 5).cdf(
                torch.tensor(1)
            )
        elif 2 < u <= 3:
            return dist.Normal(0, 5).cdf(torch.tensor(3)) - dist.Normal(0, 5).cdf(
                torch.tensor(2)
            )
        elif 3 < u <= 4:
            return dist.Normal(0, 5).cdf(torch.tensor(4)) - dist.Normal(0, 5).cdf(
                torch.tensor(3)
            )
        elif 4 < u:
            return 1.0 - dist.Normal(0, 5).cdf(torch.tensor(4))
        else:
            raise ValueError(f"Ups this should not happen. u={u}")

    def __call__(self):
        u = pyro.sample("u", dist.Normal(0, 5))
        k = self.get_k(u)
        x = pyro.sample(f"x_{k}", dist.Normal(k, self.prior_std))
        y = pyro.sample(
            "y",
            dist.Normal(x, self.likelihood_std),
            obs=torch.tensor(self.observed_data),
        )
        return k

    def calculate_ground_truth_weights(self, sdvi) -> tuple[dict[str, float], float]:
        marginal_likelihoods = dict()
        overall_Z = 0

        us = {f"u,x_{self.get_k(u)}": u for u in torch.arange(-4.5, 5.5, 1.0)}
        for slp_identifier, u_value in us.items():
            marginal_likelihoods[slp_identifier] = marginal_likelihood(
                self.observed_data,
                self.likelihood_std,
                self.get_k(u_value),
                self.prior_std,
            )
            marginal_likelihoods[slp_identifier] *= self.prior_branch_prob(u_value)
            overall_Z += marginal_likelihoods[slp_identifier]

        # Return the normalized weights so that they sum to one.
        return {
            bt: ml.item() / overall_Z for bt, ml in marginal_likelihoods.items()
        }, overall_Z

    def make_parameter_plots(self, results, guide, address_trace, file_prefix):
        # All address traces have the form "u,x_{prior_mean}"
        prior_mean = int(address_trace.split(",")[1].split("_")[1])

        post_mean, post_std = posterior_params(
            self.observed_data,
            self.likelihood_std,
            prior_mean,
            self.prior_std,
        )

        fig, axs = plt.subplots(2, 2)
        for i in range(2):
            axs[i, 0].plot([x[i] for x in results["loc"]])
            axs[i, 0].set_title(f"loc[{i}]")

            axs[i, 1].plot([math.exp(x[i]) for x in results["log_scale"]])
            axs[i, 1].set_title(f"scale[{i}]")

            if i == 1:
                axs[i, 0].axhline(
                    post_mean,
                    linestyle="--",
                    color="black",
                )
                axs[i, 1].axhline(
                    post_std,
                    linestyle="--",
                    color="black",
                )

        fig.tight_layout()
        fig.savefig(f"{file_prefix}_params.jpg")

        # Plot final distributions.
        fig, axs = plt.subplots(2, 1)
        for i in range(2):
            final_mean = results["loc"][-1][i]
            final_std = math.exp(results["log_scale"][-1][i])
            final_marginal_dist = dist.Normal(final_mean, final_std)

            if i == 0:
                min_x = final_mean - 5 * final_std
                max_x = final_mean + 5 * final_std
                xs = torch.linspace(min_x, max_x, 1000)
                axs[i].plot(xs, final_marginal_dist.log_prob(xs).exp())
            else:
                min_x = min(
                    final_mean - 5 * final_std,
                    post_mean - 5 * post_std,
                )
                max_x = max(
                    final_mean + 5 * final_std,
                    post_mean + 5 * post_std,
                )
                xs = torch.linspace(min_x, max_x, 1000)
                axs[i].plot(xs, final_marginal_dist.log_prob(xs).exp())
                axs[i].plot(
                    xs,
                    dist.Normal(post_mean, post_std).log_prob(xs).exp(),
                )

        fig.tight_layout()
        fig.savefig(f"{file_prefix}_final_marginals.jpg")
        plt.close()

    def plot_posterior_samples(self, posterior_samples, fname):
        logging.info("Not doing a posterior plot for this model.")

    def evaluation(self, posterior_samples, ground_truth_weights=None):
        return torch.tensor(float("nan"))