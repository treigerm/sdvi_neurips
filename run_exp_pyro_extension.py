import logging
import hydra
import pyro
import torch
import numpy as np
import os
import copy
import pickle
import csv
import functools
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from pyro.infer.autoguide import initialization

from pyro import param, poutine


def get_init_fn(fn_name: str):
    return getattr(initialization, fn_name)


def make_metric_plots(
    metrics,
    file_prefix,
    metric_names=["losses", "smoothed_elbos", "true_elbos"],
    plot_unique=True,
    logscale=True,
):
    for name, metric_values in metrics.items():
        if not (name in metric_names):
            continue

        if plot_unique:
            # Get unique values to remove all the repeated values we added to make
            # all the arrays the same length. Metric arrays have different lengths for
            # each SLP because each SLP might not be selected the same number of times.
            _, unique_ixs = np.unique(metric_values, return_index=True)
            metric_values = np.array(metric_values)[np.sort(unique_ixs)]
        num_values = len(metric_values)

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(metric_values)
        axs[0].set_title(f"{name}")
        axs[0].set_xlabel("Iteration")
        if logscale:
            axs[0].set_yscale("symlog")

        # Have another plot which plots the metrics after the initial "burn-in" phase.
        start_ix = 2 * int(num_values / 5)
        axs[1].plot(range(start_ix, num_values), metric_values[start_ix:])

        fig.savefig(f"{file_prefix}_{name}.jpg")
        plt.close(fig)


def plot_posterior_samples(posterior_samples, observed_data, fname):
    posterior_predictive_samples = torch.cat(
        [trace.nodes["obs"]["value"] for trace in posterior_samples]
    )

    posterior_predictive_density = gaussian_kde(
        posterior_predictive_samples, bw_method=0.05
    )
    data_density = gaussian_kde(observed_data, bw_method=0.05)

    xs = torch.linspace(-5, 30, 1000)

    fig, ax = plt.subplots()
    ax.plot(xs, posterior_predictive_density(xs), label="Post Pred")
    ax.plot(xs, data_density(xs), label="Data")

    ax.scatter(
        observed_data,
        -torch.ones(observed_data.shape[0]) * 0.01,
        alpha=0.3,
        marker="x",
        color="black",
    )
    ax.scatter(
        posterior_predictive_samples,
        -torch.ones(posterior_predictive_samples.shape[0]) * 0.01,
        alpha=0.3,
        marker="x",
        color="red",
    )
    ax.legend()
    fig.savefig(fname)
    plt.close(fig)


def plot_all_local_elbos_and_iwaes(exclusive_kl_results, top5_slp_bts):
    metric_names = ["true_elbos", "iwaes"]
    for metric_name in metric_names:
        fig, ax = plt.subplots(figsize=(20, 20))
        for bt, results in exclusive_kl_results.items():
            values = results[metric_name]

            if bt in top5_slp_bts:
                ax.plot(values, label=bt)
            else:
                ax.plot(values, color="black")

        ax.legend()
        ax.set_title(metric_name)
        ax.set_xlabel("Iteration")
        ax.set_yscale("symlog")
        fig.savefig(f"local_{metric_name}.jpg")
        plt.close(fig)


def plot_slp_weights(slp_weights, fname, ground_truth_slp_weights=None):
    # Input: Dict bt => [weight]

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    for bt, weights in slp_weights.items():
        ax.plot(weights, label=bt)
        if not (ground_truth_slp_weights is None):
            ax.axhline(ground_truth_slp_weights[bt], linestyle="--", color="black")

    ax.legend()
    fig.savefig(fname)
    plt.close(fig)


def make_elbo_plot(elbos, fname, marginal_likelihood=None):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(elbos)
    if not (marginal_likelihood is None):
        axs[0].axhline(
            torch.log(marginal_likelihood),
            linestyle="--",
            color="black",
        )
    axs[0].set_ylabel("ELBO")
    axs[0].set_xlabel("Iteration")
    axs[0].set_yscale("symlog")

    num_values = len(elbos)
    start_ix = 2 * int(num_values / 5)
    axs[1].plot(range(start_ix, num_values), elbos[start_ix:])
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def save_forward_kl_metrics(forward_kl_results, fname):
    metric_names_whitelist = ["losses"]
    metrics_to_save = dict()
    num_iterations = None
    for bt, metrics in forward_kl_results.items():
        for key, values in metrics.items():
            if not (key in metric_names_whitelist):
                continue
            new_key = f"{key}_{bt}"
            metrics_to_save[new_key] = values
            num_iterations = len(values)

    metrics_to_save["iteration"] = torch.arange(num_iterations)

    # Convert elements from torch.tensor to floats
    for k in metrics_to_save.keys():
        # Convert each element in list to a float.
        metrics_to_save[k] = [v.item() for v in metrics_to_save[k]]

    with open(fname, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics_to_save.keys())
        writer.writerows(zip(*metrics_to_save.values()))


def save_metrics(
    vi_metrics,
    global_elbos,
    log_post_densities,
    branch_weights,
    fname,
):
    metrics_to_save = {
        "iteration": torch.arange(len(global_elbos)),
        "global_elbos": global_elbos,
    }
    for bt, metrics_bt in vi_metrics.items():
        for key in metrics_bt.keys():
            new_key = f"{key}_{bt}"
            metrics_to_save[new_key] = metrics_bt[key]

        weight_key = f"weight_{bt}"
        metrics_to_save[weight_key] = branch_weights[bt]

    # Convert elements from torch.tensor to floats
    for k in metrics_to_save.keys():
        # Convert each element in list to a float.
        metrics_to_save[k] = [v.item() for v in metrics_to_save[k]]

    metrics_to_save["lppds"] = log_post_densities

    with open(fname, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics_to_save.keys())
        writer.writerows(zip(*metrics_to_save.values()))


def calculate_intermediate_lppd(
    model,
    guides,
    parameters,
    branching_sample_values,
    bt2weights,
    num_posterior_samples,
):
    guide_state_dicts = dict()
    guides = copy.deepcopy(guides)
    for bt in guides.keys():
        guide_state_dicts[bt] = guides[bt].state_dict()

    unconditioned_model = poutine.uncondition(model)

    log_posterior_predictive_densities = []

    num_iterations = len(list(bt2weights.values())[0])
    for ix in range(num_iterations):
        # Create distribution over branching traces
        bt_and_weights = list(bt2weights.items())
        branching_traces = [bt for bt, _ in bt_and_weights]
        slp_dist = pyro.distributions.Categorical(
            torch.tensor([weight[ix] for _, weight in bt_and_weights])
        )

        posterior_samples = []
        for _ in range(num_posterior_samples):
            # Sample a branching trace
            slp_ix = slp_dist.sample()
            bt = branching_traces[slp_ix]

            # Condition on all the sampled values
            slp_model = poutine.condition(
                unconditioned_model, data=branching_sample_values[bt]
            )

            # Load guide
            guide_state_dicts[bt].update(
                {k: torch.tensor(params[ix]) for k, params in parameters[bt].items()}
            )
            guide = guides[bt]
            guide.load_state_dict(guide_state_dicts[bt])

            with torch.no_grad():
                # Sample from the guide
                guide_trace = poutine.trace(guide).get_trace()

                # Replay the model
                posterior_sample = poutine.trace(
                    poutine.replay(slp_model, trace=guide_trace)
                ).get_trace()
            posterior_samples.append(posterior_sample)

        log_posterior_predictive_densities.append(
            model.evaluation(posterior_samples).item()
        )

    return log_posterior_predictive_densities


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


def forward_kl_callback(sdvi, forward_kl_results, model):
    for bt, result in forward_kl_results.items():
        os.makedirs("forward_kl_plots", exist_ok=True)
        plots_prefix = os.path.join("forward_kl_plots", f"slp{bt}_")
        make_metric_plots(
            result,
            plots_prefix,
            metric_names=["losses"],
            plot_unique=False,
            logscale=False,
        )
        model.make_parameter_plots(result, sdvi.guides[bt], bt, plots_prefix)

    save_forward_kl_metrics(forward_kl_results, "forward_kl_results.csv")


@hydra.main(config_path="conf_pyro_extension", config_name="config")
def main(cfg):
    pyro.set_rng_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)

    logging.info(os.getcwd())

    model = hydra.utils.instantiate(cfg.model)

    resource_allocation_utility = hydra.utils.instantiate(cfg.resource_allocation)
    sdvi = hydra.utils.instantiate(
        cfg.sdvi,
        model=model,
        utility_class=resource_allocation_utility,
        autoguide_hide_vars=model.autoguide_hide_vars,
        init_loc_fn=get_init_fn(cfg.init_loc_fn),
        slps_identified_by_discrete_samples=model.slps_identified_by_discrete_samples,
    )

    _, exclusive_kl_results, resource_allocation_metrics = sdvi.run(
        forward_kl_callback
    )

    # Plot diagnostics
    logging.info(resource_allocation_metrics["bt2num_selected"])
    for bt in sdvi.branching_traces:
        slp_folder = os.path.join("exclusive_kl_plots", f"slp_{bt}")
        os.makedirs(slp_folder, exist_ok=True)
        plots_prefix = os.path.join(slp_folder, f"slp{bt}_")
        make_metric_plots(
            exclusive_kl_results[bt],
            plots_prefix,
            metric_names=[
                "losses",
                "smoothed_elbos",
                "true_elbos",
                "iwaes",
                "weight_variances",
                "loc_grad_norm",
                "log_scale_grad_norm",
            ],
        )

        model.make_parameter_plots(
            exclusive_kl_results[bt], sdvi.guides[bt], bt, plots_prefix
        )

    # Make plots showing the evolution of the weights.
    branch_weights, global_elbos = sdvi.calculate_slp_weights()
    ground_truth_branch_weights, global_Z = model.calculate_ground_truth_weights(sdvi)
    plot_slp_weights(
        branch_weights,
        "slp_weights.jpg",
        ground_truth_slp_weights=ground_truth_branch_weights,
    )

    top_5_slps = [
        (k, v[-1].item())
        for k, v in sorted(
            branch_weights.items(), key=lambda item: item[1][-1], reverse=True
        )[:5]
    ]
    logging.info(f"Top 5 SLPs: {top_5_slps}")

    plot_all_local_elbos_and_iwaes(exclusive_kl_results, [x[0] for x in top_5_slps])
    # Plot evolution of the utilities for each SLP
    if "utilities" in resource_allocation_metrics:
        plot_slp_weights(resource_allocation_metrics["utilities"], "utilities.jpg")

    parameter_names = {
        bt: [n for n, _ in sdvi.guides[bt].named_parameters()]
        for bt in sdvi.branching_traces
    }
    if model.does_lppd_evaluation:
        parameters = {
            bt: {param_name: m[param_name] for param_name in parameter_names[bt]}
            for bt, m in exclusive_kl_results.items()
        }
        lppds = calculate_intermediate_lppd(
            model,
            sdvi.guides,
            parameters,
            sdvi.branching_sample_values,
            sdvi.bt2weight,
            cfg.posterior_predictive_num_samples,
        )
        plot_lppd(lppds, "lppd.jpg")
        logging.info(f"Log posterior predictive density: {lppds[-1]:.2f}")
    else:
        num_iterations = len(list(sdvi.bt2weight.values())[0])
        lppds = num_iterations * [float("nan")]

    posterior_samples = sdvi.sample_posterior_predictive(
        cfg.posterior_predictive_num_samples
    )
    model.plot_posterior_samples(posterior_samples, "posterior_predictive.jpg")

    # Make plots showing the evolution of the global ELBO
    make_elbo_plot(global_elbos, "global_elbos.jpg", marginal_likelihood=global_Z)
    logging.info(f"Global ELBO: {global_elbos[-1]}")

    # Create dict without parameters
    exclusive_kl_metrics = {
        bt: {k: v for k, v in metrics.items() if not (k in parameter_names[bt])}
        for bt, metrics in exclusive_kl_results.items()
    }
    # Save metrics into a csv file
    # Columns: iteration, global_elbo, branch_weight_{bt_identifier}, {loss_metric}_{bt_identifier}
    save_metrics(
        exclusive_kl_metrics,
        global_elbos,
        lppds,
        branch_weights,
        "exclusive_kl_results.csv",
    )

    with open("sdvi.pickle", "wb") as f:
        pickle.dump(sdvi, f)

    with open("resource_allocation_metrics.pickle", "wb") as f:
        pickle.dump(resource_allocation_metrics, f)


if __name__ == "__main__":
    main()