from typing import Union
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
import logging
import tqdm
import os
import csv
import hydra
import matplotlib.pyplot as plt

from models.pyro_extensions.dcc import DCC, SLPInfo
from models.pyro_extensions.resource_allocation import DCCUtility
from models.normal_model import posterior_params, marginal_likelihood


class NormalModel:
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

    def calculate_ground_truth_weights(self) -> tuple[dict[str, float], float]:
        marginal_likelihoods = dict()
        overall_Z = 0

        us = {str(self.get_k(u)): u for u in torch.arange(-4.5, 5.5, 1.0)}
        for slp_identifier, u_value in us.items():
            marginal_likelihoods[slp_identifier] = marginal_likelihood(
                self.observed_data,
                self.likelihood_std,
                int(slp_identifier),
                self.prior_std,
            )
            marginal_likelihoods[slp_identifier] *= self.prior_branch_prob(u_value)
            overall_Z += marginal_likelihoods[slp_identifier].item()

        # Return the normalized weights so that they sum to one.
        return {
            bt: ml.item() / overall_Z for bt, ml in marginal_likelihoods.items()
        }, overall_Z


class NormalModelGuide(NormalModel):
    def __call__(self):
        u_loc, u_scale = pyro.param("u_loc", lambda: torch.tensor(0.0)), pyro.param(
            "u_scale", lambda: torch.tensor(1.0), constraints.positive
        )
        x_params = dict()
        for i in range(10):
            x_params[f"x_{i}_loc"] = pyro.param(f"x_{i}_loc", lambda: torch.tensor(0.0))
            x_params[f"x_{i}_scale"] = pyro.param(
                f"x_{i}_scale", lambda: torch.tensor(1.0), constraints.positive
            )

        u = pyro.sample("u", dist.Normal(u_loc, u_scale))
        k = self.get_k(u)
        x = pyro.sample(
            f"x_{k}", dist.Normal(x_params[f"x_{k}_loc"], x_params[f"x_{k}_scale"])
        )
        return k


class AutoNormalScoreMessenger(pyro.infer.autoguide.AutoNormalMessenger):
    def get_posterior(
        self, name: str, prior: dist.Distribution
    ) -> Union[dist.Distribution, torch.Tensor]:
        posterior = super().get_posterior(name, prior)
        if isinstance(posterior, torch.Tensor):
            return posterior

        posterior.base_dist.has_rsample = False
        return posterior


def extract_slp_weights_vi(
    samples: list[poutine.Trace],
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    slp_weights_per_iteration = {str(i): [] for i in range(10)}
    counts = {str(i): 0 for i in range(10)}
    total_count = 0

    samples_by_address = {f"x_{i}": [] for i in range(10)}
    samples_by_address["u"] = []
    for sample in samples:
        samples_by_address["u"].append(sample.nodes["u"]["value"])
        for i in range(10):
            address = f"x_{i}"
            if address in sample.nodes.keys():
                counts[str(i)] += 1
                total_count += 1
                samples_by_address[address].append(sample.nodes[address]["value"])
                for slp_identifier, slp_count in counts.items():
                    slp_weights_per_iteration[slp_identifier].append(
                        slp_count / total_count
                    )
                break

    samples_by_address = {k: torch.tensor(v) for k, v in samples_by_address.items()}

    return slp_weights_per_iteration, samples_by_address


def extract_slp_weights_dcc(
    slps_info: dict[str, SLPInfo],
    marginal_likelihood_per_iteration: dict[str, list[torch.Tensor]],
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    num_iterations = len(list(marginal_likelihood_per_iteration.values())[0])
    assert all(
        [len(v) == num_iterations for v in marginal_likelihood_per_iteration.values()]
    )

    slp_weights: dict[str, list[float]] = {
        k[-1]: [] for k in marginal_likelihood_per_iteration.keys()
    }
    for ix in range(num_iterations):
        total_marginal_likelihood = sum(
            [v[ix].item() for v in marginal_likelihood_per_iteration.values()]
        )
        for k in marginal_likelihood_per_iteration.keys():
            slp_weights[k[-1]].append(
                marginal_likelihood_per_iteration[k][ix].item()
                / total_marginal_likelihood
            )

    samples_by_address = {f"x_{i}": [] for i in range(10)}
    samples_by_address["u"] = []
    for _, slp_info in slps_info.items():
        for chain in slp_info.mcmc_samples:
            for sample in chain:
                samples_by_address["u"].append(sample.nodes["u"]["value"])
                for i in range(10):
                    address = f"x_{i}"
                    if address in sample.nodes.keys():
                        samples_by_address[address].append(
                            sample.nodes[address]["value"]
                        )

    samples_by_address = {k: torch.tensor(v) for k, v in samples_by_address.items()}
    return slp_weights, samples_by_address


def get_posterior_means(
    samples: list[poutine.Trace], addresses: list[str]
) -> dict[str, torch.Tensor]:
    posterior_sums = {a: torch.tensor(0.0) for a in addresses}
    posterior_means = {
        a: torch.ones((len(samples),)).fill_(torch.tensor(float("nan")))
        for a in addresses
    }
    num_address_encountered = {a: 0 for a in addresses}

    iteration_ix = 1
    for sample in samples:
        for addr in addresses:
            if addr in sample.nodes.keys():
                value = sample.nodes[addr]["value"]
                posterior_sums[addr] = posterior_sums[addr] + value
                num_address_encountered[addr] += 1
                mean = posterior_sums[addr] / num_address_encountered[addr]
                posterior_means[addr][iteration_ix - 1] = mean
            else:
                posterior_means[addr][iteration_ix - 1] = posterior_means[addr][
                    iteration_ix - 2
                ]

        iteration_ix += 1

    return posterior_means


def plot_slp_weight_error(
    estimated_weights: dict[str, list[float]],
    ground_truth_weights: dict[str, float],
    fname: str,
):
    slp_identifiers = list(ground_truth_weights.keys())
    ground_truth_tensor = torch.stack(
        [torch.tensor(ground_truth_weights[a]) for a in slp_identifiers], dim=0
    )

    num_iterations = len(estimated_weights[slp_identifiers[0]])
    errors = torch.ones((num_iterations,))
    for ix in range(num_iterations):
        mean_tensor = torch.stack(
            [torch.tensor(estimated_weights[a][ix]) for a in slp_identifiers], dim=0
        )
        errors[ix] = torch.norm(mean_tensor - ground_truth_tensor)

    fig, ax = plt.subplots(figsize=(10, 15))
    ax.plot(errors.numpy())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 error")
    fig.savefig(fname)


def plot_posterior_mean_error(
    estimated_means: dict[str, torch.Tensor],
    ground_truth_means: dict[str, float],
    fname: str,
):
    addresses_list = list(ground_truth_means.keys())
    ground_truth_tensor = torch.stack(
        [torch.tensor(ground_truth_means[a]) for a in addresses_list], dim=0
    )

    num_iterations = estimated_means[addresses_list[0]].size(0)
    errors = torch.ones((num_iterations,))
    for ix in range(num_iterations):
        mean_tensor = torch.stack(
            [estimated_means[a][ix] for a in addresses_list], dim=0
        )
        if ix == num_iterations - 1:
            print(mean_tensor)
        errors[ix] = torch.norm(mean_tensor - ground_truth_tensor)

    fig, ax = plt.subplots(figsize=(10, 15))
    ax.plot(errors.numpy())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 error")
    fig.save0fig(fname)


@hydra.main(config_path="baselines_config", config_name="config")
def main(cfg):
    pyro.set_rng_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)

    logging.info(os.getcwd())

    model = NormalModel()

    if cfg.inference_algo._target_ == "models.pyro_extensions.dcc.DCC":
        dcc: DCC = hydra.utils.instantiate(
            cfg.inference_algo, model=model, utility=DCCUtility()
        )
        slps_info, marginal_likelihood_per_iteration = dcc.run()
        slp_weights, samples_by_address = extract_slp_weights_dcc(
            slps_info, marginal_likelihood_per_iteration
        )
    elif cfg.inference_algo._target_ == "SVI":
        guide = NormalModelGuide()
        optim = pyro.optim.Adam({"lr": cfg.inference_algo.learning_rate})
        svi = pyro.infer.SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO())

        pyro.clear_param_store()
        for _ in tqdm.tqdm(range(cfg.inference_algo.num_steps)):
            svi.step()

        posterior_samples = [
            poutine.trace(guide).get_trace()
            for _ in range(cfg.inference_algo.num_posterior_samples)
        ]
        slp_weights, samples_by_address = extract_slp_weights_vi(posterior_samples)

        u_loc = pyro.param("u_loc").item()
        u_scale = pyro.param("u_scale").item()
        print(f"u parameters: u_loc={u_loc} u_scale={u_scale}")
    elif cfg.inference_algo._target_ == "AutoNormalMessengerSVI":
        if cfg.inference_algo.gradient_estimator == "reparam":
            guide = pyro.infer.autoguide.AutoNormalMessenger(
                model, init_loc_fn=pyro.infer.autoguide.init_to_sample
            )
        elif cfg.inference_algo.gradient_estimator == "score":
            guide = AutoNormalScoreMessenger(
                model, init_loc_fn=pyro.infer.autoguide.init_to_sample
            )

        optim = pyro.optim.Adam({"lr": cfg.inference_algo.learning_rate})
        svi = pyro.infer.SVI(
            model,
            guide,
            optim,
            loss=pyro.infer.Trace_ELBO(num_particles=cfg.inference_algo.num_particles),
        )

        pyro.clear_param_store()
        slp_weights = {str(i): [] for i in range(10)}
        elbos = []
        every_k = int(cfg.inference_algo.num_steps / cfg.inference_algo.num_checkpoints)
        for i in tqdm.tqdm(range(cfg.inference_algo.num_steps)):
            svi.step()
            if i % every_k == 0:
                posterior_samples = [
                    poutine.trace(guide).get_trace()
                    for _ in range(cfg.inference_algo.num_posterior_samples)
                ]
                weights, _ = extract_slp_weights_vi(posterior_samples)
                for addr in slp_weights.keys():
                    slp_weights[addr].append(weights[addr][-1])

                # Calculate ELBO value.
                trace_elbo = pyro.infer.Trace_ELBO(
                    num_particles=cfg.inference_algo.num_posterior_samples
                )
                elbos.append(-trace_elbo.loss(model, guide))

        posterior_samples = [
            poutine.trace(guide).get_trace()
            for _ in range(cfg.inference_algo.num_posterior_samples)
        ]
        weights, samples_by_address = extract_slp_weights_vi(posterior_samples)
        for addr in slp_weights.keys():
            slp_weights[addr].append(weights[addr][-1])

        u_loc, u_scale = guide._get_params("u", dist.Normal(0, 5))
        logging.info(f"u parameters: u_loc={u_loc} u_scale={u_scale}")

        # Calculate ELBO value.
        trace_elbo = pyro.infer.Trace_ELBO(
            num_particles=cfg.inference_algo.num_posterior_samples
        )
        elbos.append(-trace_elbo.loss(model, guide))
        logging.info(f"Final ELBO: {elbos[-1]}")
        with open("elbos.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["elbos"])
            for elbo in elbos:
                writer.writerow([elbo])

    logging.info("Estimated parameters:")
    ground_truth_means = dict()
    for address, values in samples_by_address.items():
        if address == "u":
            continue

        prior_mean = int(address.split("_")[1])
        post_mean, post_std = posterior_params(
            model.observed_data,
            model.likelihood_std,
            prior_mean,
            model.prior_std,
        )
        ground_truth_means[address] = post_mean
        logging.info(
            f"{address}: {values.mean()} +/- {values.std()} ({post_mean} +/- {post_std})"
        )

    # plot_posterior_mean_error(
    #     post_means, ground_truth_means, "error_mean_estimates.jpg"
    # )
    ground_truth_weights, overall_Z = model.calculate_ground_truth_weights()
    plot_slp_weight_error(
        slp_weights, ground_truth_weights, "slp_weight_estimates_error.jpg"
    )

    logging.info("SLP weights:")
    last_slp_weights = {k: v[-1] for k, v in slp_weights.items()}
    logging.info(f"Computed: {last_slp_weights}")
    logging.info(f"Ground truth: {ground_truth_weights}")

    with open("estimated_weights.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(slp_weights.keys())
        writer.writerows(zip(*slp_weights.values()))


if __name__ == "__main__":
    main()
