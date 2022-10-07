import pickle
import math
import torch
import pyro
import os
import pyro.distributions as dist
import argparse

import sys

HOME_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(HOME_DIR)
from models.pyro_extensions.infer import SDVI
from models.pyro_extensions.resource_allocation import SuccessiveHalving


BRANCH1_PRIOR_MEAN = -3
BRANCH2_PRIOR_MEAN = 3
PRIOR_STD = 1
LIKELIHOOD_STD = 2

OBSERVED_DATA = 2


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


class ToyModel:
    observed_data = torch.tensor(OBSERVED_DATA)

    branch1_prior_mean = BRANCH1_PRIOR_MEAN
    branch2_prior_mean = BRANCH2_PRIOR_MEAN
    prior_std = PRIOR_STD
    likelihood_std = LIKELIHOOD_STD

    def __init__(self, cut_point=0.0):
        self.branch1_post_mean, self.branch1_post_std = posterior_params(
            self.observed_data,
            self.likelihood_std,
            self.branch1_prior_mean,
            self.prior_std,
        )
        self.branch1_Z = marginal_likelihood(
            self.observed_data,
            self.likelihood_std,
            self.branch1_prior_mean,
            self.prior_std,
        )
        self.branch2_post_mean, self.branch2_post_std = posterior_params(
            self.observed_data,
            self.likelihood_std,
            self.branch2_prior_mean,
            self.prior_std,
        )
        self.branch2_Z = marginal_likelihood(
            self.observed_data,
            self.likelihood_std,
            self.branch2_prior_mean,
            self.prior_std,
        )

        self.cut_point = cut_point

        z0_prior = dist.Normal(0, 1)
        self.branch1_prior = z0_prior.cdf(torch.tensor(cut_point))
        self.marginal_likelihood = (
            self.branch1_prior * self.branch1_Z
            + (1 - self.branch1_prior) * self.branch2_Z
        )

        self.branch1_post_prob = (
            self.branch1_prior * self.branch1_Z
        ) / self.marginal_likelihood

    def __call__(self):
        z0 = pyro.sample("z0", dist.Normal(0, 1))
        if z0 < self.cut_point:
            z1 = pyro.sample("z1", dist.Normal(self.branch1_prior_mean, self.prior_std))
        else:
            z1 = pyro.sample("z2", dist.Normal(self.branch2_prior_mean, self.prior_std))

        x = pyro.sample(
            "x", dist.Normal(z1, self.likelihood_std), obs=self.observed_data
        )
        return z0.item(), z1, x


class AutoNormalScoreMessenger(pyro.infer.autoguide.AutoNormalMessenger):
    def get_posterior(self, name: str, prior: dist.Distribution):
        posterior = super().get_posterior(name, prior)
        if isinstance(posterior, torch.Tensor):
            return posterior

        posterior.base_dist.has_rsample = False
        return posterior


def estimate_branch2_prob(guide):
    z0_post_dist = dist.Normal(
        guide.locs.z0_unconstrained, torch.exp(guide.scales.z0_unconstrained)
    )
    return 1 - z0_post_dist.cdf(torch.tensor(0.0)).item()


def get_branch2_weights(
    toy_model,
    num_replications=10,
    num_iterations=2000,
    guide_name="Pyro",
    learning_rate=0.01,
):
    branch2_weights = torch.zeros((num_replications, num_iterations))
    for i in range(num_replications):
        if guide_name == "Pyro":
            guide = pyro.infer.autoguide.AutoNormalMessenger(toy_model)
        elif guide_name == "BBVI":
            guide = AutoNormalScoreMessenger(toy_model)
        else:
            raise NotImplementedError()

        optim = pyro.optim.Adam({"lr": learning_rate})
        svi = pyro.infer.SVI(
            toy_model,
            guide,
            optim,
            loss=pyro.infer.Trace_ELBO(num_particles=1),
        )

        for j in range(num_iterations):
            loss = svi.step()
            branch2_weights[i, j] = estimate_branch2_prob(guide)

    return branch2_weights


def forward_kl_callback(sdvi, forward_kl_results, model):
    pass


def get_branch2_weights_sdvi(
    toy_model, num_replications=10, num_iterations=2000, save_metrics_every_n=10
):
    branch2_weights = torch.zeros(
        (num_replications, int(num_iterations / (2 * save_metrics_every_n)))
    )
    for ix in range(num_replications):
        sdvi = SDVI(
            model=toy_model,
            learning_rate=0.01,
            guide_class_name="MeanFieldNormal",
            utility_class=SuccessiveHalving(
                num_total_iterations=num_iterations, num_final_arms=2
            ),
            find_slp_samples=10,
            forward_kl_num_particles=10,
            forward_kl_iter=0,
            num_parallel_processes=2,
            save_metrics_every_n=save_metrics_every_n,
            elbo_estimate_num_particles=10,
        )
        sdvi.run(forward_kl_callback)
        sdvi_bt2weights, _ = sdvi.calculate_slp_weights()
        branch2_weights[ix, :] = sdvi_bt2weights["z0,z2"]
    return branch2_weights


def main(run_id):
    pyro.set_rng_seed(int(run_id))
    torch.manual_seed(int(run_id))

    cut_off = 0.0
    toy_model = ToyModel(cut_point=cut_off)
    num_replications = 20

    br2_weights_pyro = get_branch2_weights(
        toy_model, num_replications=num_replications, guide_name="Pyro"
    )
    br2_weights_bbvi = get_branch2_weights(
        toy_model, num_replications=num_replications, guide_name="BBVI"
    )
    br2_weights_sdvi = get_branch2_weights_sdvi(
        toy_model, num_replications=num_replications, save_metrics_every_n=1
    )

    with open(
        os.path.join(
            HOME_DIR,
            "notebooks",
            "figures",
            f"motivating_example_results_elbo_estimate_num_particles=10_cut_off={cut_off}_num_replications={num_replications}_run_id{run_id}.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(
            {
                "br2_weights_pyro": br2_weights_pyro,
                "br2_weights_bbvi": br2_weights_bbvi,
                "br2_weights_sdvi": br2_weights_sdvi,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "id",
        type=int,
    )
    args = parser.parse_args()
    main(args.id)