from collections import defaultdict
from typing import Callable

import copy
import pickle
import torch
import pyro
import numpy as np
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
import logging
import tqdm
import os
import csv
import hydra
import matplotlib.pyplot as plt

from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn.module import PyroParam
from torch.distributions import constraints

from models.real_gmm import InfiniteGMMModel
from models.pyro_extensions.dcc import DCC, SLPInfo, IterationInfo
from models.pyro_extensions.resource_allocation import DCCUtility


class BBVIAutoGuide(pyro.infer.autoguide.AutoMessenger):
    def __init__(
        self,
        model: Callable,
        *,
        init_loc_fn: Callable = pyro.infer.autoguide.init_to_mean,
        init_scale: float = 0.1,
        amortized_plates=(),
    ):
        super().__init__(model, amortized_plates=amortized_plates)
        self.init_loc_fn = init_loc_fn
        self.init_scale = init_scale

    def get_posterior(self, name: str, prior: dist.Distribution) -> dist.Distribution:
        if isinstance(prior, dist.Normal):
            loc, scale = self._get_params(name, prior)
            posterior = dist.Normal(loc, scale)
            posterior.has_rsample = False
            return posterior
        elif isinstance(prior, dist.MultivariateNormal):
            loc, scale = self._get_params(name, prior)
            posterior = dist.MultivariateNormal(loc, torch.diag(scale))
            posterior.has_rsample = False
            return posterior
        elif isinstance(prior, dist.Poisson):
            rate = self._get_params(name, prior)
            return dist.Poisson(rate)

    def _get_params(self, name: str, prior: dist.Distribution):
        try:
            if isinstance(prior, dist.Normal) or isinstance(
                prior, dist.MultivariateNormal
            ):
                loc = deep_getattr(self.locs, name)
                scale = deep_getattr(self.scales, name)
                return loc, scale
            elif isinstance(prior, dist.Poisson):
                rate = deep_getattr(self.rates, name)
                return rate
        except AttributeError:
            pass

        if isinstance(prior, dist.Normal) or isinstance(prior, dist.MultivariateNormal):
            init_loc = self.init_loc_fn({"name": name, "fn": prior}).detach()
            init_scale = torch.full_like(init_loc, self.init_scale)

            deep_setattr(self, "locs." + name, PyroParam(init_loc))
            deep_setattr(
                self,
                "scales." + name,
                PyroParam(init_scale, constraint=constraints.positive),
            )
        elif isinstance(prior, dist.Poisson):
            init_rate = prior.rate
            deep_setattr(
                self,
                "rates." + name,
                PyroParam(init_rate, constraint=constraints.nonnegative),
            )

        return self._get_params(name, prior)


class BBVIFiniteAutoGuide(pyro.infer.autoguide.AutoMessenger):
    def __init__(
        self,
        model: Callable,
        *,
        init_loc_fn: Callable = pyro.infer.autoguide.init_to_mean,
        init_scale: float = 0.1,
        upper_bound: int = 50,
        amortized_plates=(),
    ):
        super().__init__(model, amortized_plates=amortized_plates)
        self.init_loc_fn = init_loc_fn
        self.init_scale = init_scale
        self.upper_bound = upper_bound

    def get_posterior(self, name: str, prior: dist.Distribution) -> dist.Distribution:
        if isinstance(prior, dist.Normal):
            loc, scale = self._get_params(name, prior)
            posterior = dist.Normal(loc, scale)
            posterior.has_rsample = False
            return posterior
        elif isinstance(prior, dist.MultivariateNormal):
            loc, scale = self._get_params(name, prior)
            posterior = dist.MultivariateNormal(loc, torch.diag(scale))
            posterior.has_rsample = False
            return posterior
        elif isinstance(prior, dist.Poisson):
            logits = self._get_params(name, prior)
            return dist.Categorical(logits=logits)

    def _get_params(self, name: str, prior: dist.Distribution):
        try:
            if isinstance(prior, dist.Normal) or isinstance(
                prior, dist.MultivariateNormal
            ):
                loc = deep_getattr(self.locs, name)
                scale = deep_getattr(self.scales, name)
                return loc, scale
            elif isinstance(prior, dist.Poisson):
                logits = deep_getattr(self.logits, name)
                return logits
        except AttributeError:
            pass

        if isinstance(prior, dist.Normal) or isinstance(prior, dist.MultivariateNormal):
            init_loc = self.init_loc_fn({"name": name, "fn": prior}).detach()
            init_scale = torch.full_like(init_loc, self.init_scale)

            deep_setattr(self, "locs." + name, PyroParam(init_loc))
            deep_setattr(
                self,
                "scales." + name,
                PyroParam(init_scale, constraint=constraints.positive),
            )
        elif isinstance(prior, dist.Poisson):
            logits = torch.ones((self.upper_bound))
            deep_setattr(
                self,
                "logits." + name,
                PyroParam(logits),
            )

        return self._get_params(name, prior)


def extract_slp_weights_dcc(
    slps_info,
    log_marginal_likelihood_per_iteration: dict[str, list[torch.Tensor]],
) -> dict[str, list[float]]:
    num_iterations = len(list(log_marginal_likelihood_per_iteration.values())[0])
    assert all(
        [
            len(v) == num_iterations
            for v in log_marginal_likelihood_per_iteration.values()
        ]
    )

    slp_weights: dict[str, list[float]] = {
        k: [] for k in log_marginal_likelihood_per_iteration.keys()
    }
    for ix in range(num_iterations):
        normalizer = torch.logsumexp(
            torch.tensor(
                [v[ix].item() for v in log_marginal_likelihood_per_iteration.values()]
            ),
            dim=0,
        )
        for k in log_marginal_likelihood_per_iteration.keys():
            if normalizer > torch.tensor(float("-inf")):
                slp_weights[k].append(
                    (log_marginal_likelihood_per_iteration[k][ix].item() - normalizer)
                    .exp()
                    .item()
                )
            else:
                slp_weights[k].append(
                    1 / len(log_marginal_likelihood_per_iteration.keys())
                )

    return slp_weights


def get_mc_estimate(
    fun: Callable,
    num_selected: int,
    slps_info: SLPInfo,
    num_steps_per_iteration: int = 1,
) -> torch.Tensor:
    fun_evals = []
    for chain in slps_info.mcmc_samples:
        for trace in chain[: num_selected * num_steps_per_iteration]:
            fun_evals.append(fun(trace))

    return torch.tensor(fun_evals).mean().item()


def get_iteration_mc_estimate(
    fun: Callable,
    num_selected: int,
    slps_info: SLPInfo,
    num_steps_per_iteration: int,
) -> torch.Tensor:
    fun_evals = []
    for chain in slps_info.mcmc_samples:
        for trace in chain[
            (num_selected - 1)
            * num_steps_per_iteration : num_selected
            * num_steps_per_iteration
        ]:
            fun_evals.append(fun(trace))

    return torch.tensor(fun_evals).mean().item()


def posterior_expectations(
    fun: Callable,
    slps_info: dict[str, SLPInfo],
    iteration_info: IterationInfo,
    num_steps_per_iteration: int = 1,
) -> list[float]:
    num_iterations = len(
        list(iteration_info.log_marginal_likelihoods_per_iteration.values())[0]
    )

    selected_per_iteration: list[dict(str, int)] = [{at: 1 for at in slps_info.keys()}]
    for addr_trace in iteration_info.selected_addresses:
        old_count = copy.deepcopy(selected_per_iteration[-1])
        old_count[addr_trace] += 1
        selected_per_iteration.append(old_count)

    slp_weights = extract_slp_weights_dcc(
        slps_info, iteration_info.log_marginal_likelihoods_per_iteration
    )
    num_slps = len(slp_weights.keys())

    # Create this dict to avoid potential nondeterministic ordering when calling
    # slp_weights.keys().
    at2ix = {at: ix for ix, at in enumerate(slp_weights.keys())}

    # Estimate after initialization
    local_estimates = torch.zeros(num_slps)
    for addr_trace, ix in at2ix.items():
        local_mc_estimate = get_mc_estimate(
            fun,
            selected_per_iteration[0][addr_trace],
            slps_info[addr_trace],
            num_steps_per_iteration,
        )
        local_estimates[ix] = slp_weights[addr_trace][0] * local_mc_estimate

    expectation_estimates = [local_estimates.sum().item()]
    for iteration_ix, selected_addr_trace in enumerate(
        iteration_info.selected_addresses
    ):
        iteration_mc_estimate = get_iteration_mc_estimate(
            fun,
            selected_per_iteration[iteration_ix][selected_addr_trace],
            slps_info[selected_addr_trace],
            num_steps_per_iteration,
        )
        selected_addr_trace_ix = at2ix[selected_addr_trace]
        local_estimates[selected_addr_trace_ix] = 0.5 * (
            local_estimates[selected_addr_trace_ix]
            + slp_weights[selected_addr_trace][iteration_ix] * iteration_mc_estimate
        )
        expectation_estimates.append(local_estimates.sum().item())

    return expectation_estimates


def lppd_evaluation(model, slps_info, iteration_info, num_steps_per_iteration):
    with np.load(model.validation_data_path) as d:
        validation_data = torch.tensor(d["obs"])

    def eval_fun(sample_trace):
        trace = poutine.trace(poutine.replay(model, trace=sample_trace)).get_trace()
        predictive_dist = trace.nodes["obs"]["fn"]
        # The validation data only contains half of the number of data points
        # as the training dataset.
        return predictive_dist.log_prob(torch.cat([validation_data, validation_data]))[
            : int(validation_data.size(0) / 2)
        ].sum()

    return posterior_expectations(
        eval_fun, slps_info, iteration_info, num_steps_per_iteration
    )


def plot_lppd(log_posterior_predictive_densities, fname):
    fig, axs = plt.subplots(1, 2)
    num_values = len(log_posterior_predictive_densities)
    axs[0].plot(log_posterior_predictive_densities)
    axs[0].set_yscale("symlog")

    start_ix = 2 * int(num_values / 5)
    axs[1].plot(
        range(start_ix, num_values), log_posterior_predictive_densities[start_ix:]
    )
    fig.tight_layout()
    fig.savefig(fname)


def dcc_hide_fn(msg):
    return msg["name"] in ["obs", "_RETURN"]


@hydra.main(config_path="gmm_baselines_conf", config_name="config")
def main(cfg):
    pyro.set_rng_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)

    logging.info(os.getcwd())

    model = InfiniteGMMModel(
        data_path=hydra.utils.to_absolute_path(
            "data/gmm_higher_dims/gmm_data.npz"
        ),
        validation_data_path=hydra.utils.to_absolute_path(
            "data/gmm_higher_dims/gmm_data_validation.npz"
        ),
        cluster_means_dim=100,
        num_observations=1000,
    )

    if cfg.inference_algo._target_.startswith("BBVI"):
        if cfg.inference_algo._target_ == "BBVI":
            guide = BBVIAutoGuide(model)
        elif cfg.inference_algo._target_ == "BBVIFinite":
            guide = BBVIFiniteAutoGuide(
                model, upper_bound=cfg.inference_algo.upper_bound
            )
        optim = pyro.optim.Adam({"lr": cfg.inference_algo.learning_rate})
        svi = pyro.infer.SVI(
            model,
            guide,
            optim,
            loss=pyro.infer.Trace_ELBO(
                num_particles=cfg.inference_algo.num_elbo_particles
            ),
        )
        lppds = []
        cluster_probs = []
        elbos = []
        for i in tqdm.tqdm(range(cfg.inference_algo.num_iterations)):
            svi.step()
            if (i + 1) % cfg.inference_algo.evaluate_every_n == 0 or i == 0:
                # Get unconditioned model
                unconditioned_model = pyro.poutine.uncondition(model)

                posterior_samples = []
                cluster_ps = defaultdict(float)
                elbo = torch.tensor(0.0)
                for _ in range(cfg.inference_algo.num_posterior_samples):
                    # Sample posterior values
                    guide()
                    model_trace, q_trace = guide.get_traces()

                    posterior_samples.append(
                        pyro.poutine.trace(
                            pyro.poutine.replay(unconditioned_model, trace=q_trace)
                        ).get_trace()
                    )

                    elbo += (model_trace.log_prob_sum() - q_trace.log_prob_sum()) / (
                        cfg.inference_algo.num_posterior_samples
                    )

                    num_clusters = int(q_trace.nodes["num_clusters"]["value"].item())
                    cluster_ps[num_clusters] += (
                        1 / cfg.inference_algo.num_posterior_samples
                    )

                cluster_probs.append(cluster_ps)
                lppds.append(model.evaluation(posterior_samples).item())
                elbos.append(elbo.item())

        with open("elbos.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["elbos"])
            for elbo in elbos:
                writer.writerow([elbo])

        # Save the guide.
        with open("guide.pickle", "wb") as f:
            pickle.dump(guide, f)

    elif cfg.inference_algo._target_ == "models.pyro_extensions.dcc.DCC":
        dcc = hydra.utils.instantiate(
            cfg.inference_algo,
            model=model,
            utility=DCCUtility(),
            mcmc_sample_hide_fn=dcc_hide_fn,
        )
        slps_info, iteration_info = dcc.run()
        logging.info("Starting lppd evaluation")
        lppds = lppd_evaluation(
            model,
            slps_info,
            iteration_info,
            cfg.inference_algo.num_mcmc_steps_per_iteration,
        )
        logging.info("Finished lppd evaluation")

        slp_weights = extract_slp_weights_dcc(
            slps_info, iteration_info.log_marginal_likelihoods_per_iteration
        )
        cluster_probs = {at: ws[-1] for at, ws in slp_weights.items()}

    logging.info(cluster_probs)

    plot_lppd(lppds, "lppds.jpg")

    with open("lppds.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["lppds"])
        for lppd in lppds:
            writer.writerow([lppd])

    with open("cluster_probs.pickle", "wb") as f:
        pickle.dump(cluster_probs, f)


if __name__ == "__main__":
    main()