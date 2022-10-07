import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import logging
import copy

from sklearn.decomposition import PCA

from .abstract_model import AbstractModel

MAX_NUM_CLUSTERS = 10
NUM_OBSERVATIONS = 100


class FiniteGMMModel(AbstractModel):
    max_num_clusters = MAX_NUM_CLUSTERS
    slps_identified_by_discrete_samples = True

    autoguide_hide_vars = ["num_clusters"]

    does_lppd_evaluation = True

    def __init__(
        self,
        data_path=None,
        validation_data_path=None,
        cluster_means_dim=2,
        num_observations=NUM_OBSERVATIONS,
        ordered_cluster_means=False,
        cluster_means_prior_std=10,
    ):
        self.observed_data = None
        self.ground_truth = None
        self.validation_data_path = validation_data_path
        self.cluster_means_dim = cluster_means_dim
        self.num_observations = num_observations
        self.ordered_cluster_means = ordered_cluster_means
        self.cluster_means_prior_std = cluster_means_prior_std

        if not (data_path is None):
            with np.load(data_path) as d:
                self.observed_data = torch.tensor(d["obs"])
                self.ground_truth = {k: d[k] for k in d.files}

                logging.info("Ground truth")
                logging.info(20 * "=" + "\n")
                num_clusters = self.ground_truth["num_clusters"]
                logging.info(f"Number of clusters: {num_clusters}")
                for i in range(num_clusters):
                    m = self.ground_truth[f"mean_{i}"]
                    logging.info(f"Mean {i}: {m}")

    def __call__(self):
        return self.model()

    def model(self):
        num_clusters = (
            pyro.sample(
                "num_clusters",
                dist.Categorical(
                    probs=torch.ones(self.max_num_clusters) / self.max_num_clusters
                ),
                infer={"branching": True},
            )
            + 1
        )
        num_clusters = int(num_clusters.item())
        cluster_means = torch.zeros((num_clusters, self.cluster_means_dim))
        for k in range(num_clusters):
            cluster_mean = torch.zeros(self.cluster_means_dim)
            cluster_std = self.cluster_means_prior_std * torch.eye(
                self.cluster_means_dim
            )
            if self.ordered_cluster_means:
                # Ensure that the clusters in the first dimension are evenly distributed between -5 and 5.
                cluster_mean[0] = (10 * ((k + 1) / (num_clusters + 1))) - 5
                cluster_std[0, 0] = 10 / (num_clusters + 1)

            cluster_means[k] = pyro.sample(
                f"mean_{k}",
                dist.MultivariateNormal(cluster_mean, cluster_std),
            )

        with pyro.plate("data", self.num_observations):
            mix = dist.Categorical(probs=(torch.ones(num_clusters) / num_clusters))
            comp = dist.Independent(dist.Normal(cluster_means, 0.1), 1)
            obs = pyro.sample(
                "obs", dist.MixtureSameFamily(mix, comp), obs=self.observed_data
            )

        return cluster_means, obs

    def make_parameter_plots(self, results, guide, branching_trace, file_prefix):
        # Make plot showing the evolution of the mean+scale for each cluster
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        guide = copy.deepcopy(guide)
        state_dict = guide.state_dict()

        cluster_means = results["loc"]

        # Plot evolution of parameters
        for i in range(len(cluster_means[0])):
            ax.plot([x[i] for x in cluster_means])
            ax.set_title("Means")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Posterior means")

        ground_truth_num_clusters = self.ground_truth["num_clusters"]
        ground_truth_cluster_means = np.concatenate(
            [self.ground_truth[f"mean_{i}"] for i in range(ground_truth_num_clusters)]
        )
        for i in range(ground_truth_cluster_means.shape[0]):
            ax.axhline(ground_truth_cluster_means[i], linestyle="--", color="black")

        fig.tight_layout()
        fig.savefig(f"{file_prefix}_params.jpg")

        # Plot locations of final cluster means
        final_cluster_means = np.array(cluster_means[-1])
        final_cluster_means = np.vstack(
            np.split(
                final_cluster_means,
                len(final_cluster_means) / self.cluster_means_dim,
            )
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        observed_data = copy.copy(self.observed_data)
        if self.cluster_means_dim > 2:
            # Do dimensionality reduction
            pca = PCA(n_components=2)
            pca.fit(observed_data)
            observed_data = pca.transform(observed_data)
            final_cluster_means = pca.transform(final_cluster_means)

        ax.scatter(
            observed_data[:, 0],
            observed_data[:, 1],
            alpha=0.1,
            color="black",
            label="Data",
        )
        ax.scatter(
            final_cluster_means[:, 0],
            final_cluster_means[:, 1],
            color="red",
            label="Posterior Cluster Means",
            marker="x",
        )
        ax.legend()
        fig.savefig(f"{file_prefix}_cluster_means.jpg")

        plt.close()

    def evaluation(self, posterior_samples, ground_truth_weights=None):
        # Load validation data.
        with np.load(self.validation_data_path) as d:
            validation_data = torch.tensor(d["obs"])

        log_posterior_densities = torch.zeros((len(posterior_samples)))
        for i, trace in enumerate(posterior_samples):
            predictive_dist = trace.nodes["obs"]["fn"]
            # The validation data only contains half of the number of data points
            # as the training dataset.
            log_posterior_densities[i] = predictive_dist.log_prob(
                torch.cat([validation_data, validation_data])
            )[: int(validation_data.size(0) / 2)].sum()

        lppd = torch.mean(log_posterior_densities)
        return lppd

    def plot_posterior_samples(self, posterior_samples, fname):
        posterior_predictive_samples = torch.cat(
            [trace.nodes["obs"]["value"] for trace in posterior_samples]
        )

        observed_data = self.observed_data
        if self.cluster_means_dim > 2:
            # Do dimensionality reduction
            pca = PCA(n_components=2)
            pca.fit(observed_data)
            posterior_predictive_samples = pca.transform(
                posterior_predictive_samples.numpy()
            )
            observed_data = pca.transform(observed_data)

        fig, ax = plt.subplots()

        ax.scatter(
            posterior_predictive_samples[:, 0],
            posterior_predictive_samples[:, 1],
            alpha=0.1,
            color="red",
            label="Posterior Samples",
        )
        ax.scatter(
            observed_data[:, 0],
            observed_data[:, 1],
            alpha=0.3,
            color="black",
            label="Data",
        )
        ax.legend()
        fig.savefig(fname)


class InfiniteGMMModel(FiniteGMMModel):

    max_num_clusters = float("inf")
    poisson_rate = 9

    def __init__(
        self,
        data_path=None,
        validation_data_path=None,
        cluster_means_dim=2,
        num_observations=NUM_OBSERVATIONS,
        cluster_means_prior_std=10,
    ):
        self.observed_data = None
        self.ground_truth = None
        self.validation_data_path = validation_data_path
        self.cluster_means_dim = cluster_means_dim
        self.num_observations = num_observations
        self.cluster_means_prior_std = cluster_means_prior_std

        if not (data_path is None):
            with np.load(data_path) as d:
                self.observed_data = torch.tensor(d["obs"])
                self.ground_truth = {k: d[k] for k in d.files}

                logging.info("Ground truth")
                logging.info(20 * "=" + "\n")
                num_clusters = self.ground_truth["num_clusters"]
                logging.info(f"Number of clusters: {num_clusters}")
                for i in range(num_clusters):
                    m = self.ground_truth[f"mean_{i}"]
                    logging.info(f"Mean {i}: {m}")

    def model(self):
        num_clusters = (
            pyro.sample(
                "num_clusters",
                dist.Poisson(self.poisson_rate),
                infer={"branching": True},
            )
            + 1
        )
        num_clusters = int(num_clusters.item())
        cluster_means = torch.zeros((num_clusters, self.cluster_means_dim))
        for k in range(num_clusters):
            cluster_mean = torch.zeros(self.cluster_means_dim)
            cluster_std = self.cluster_means_prior_std * torch.eye(
                self.cluster_means_dim
            )
            cluster_means[k] = pyro.sample(
                f"mean_{k}",
                dist.MultivariateNormal(cluster_mean, cluster_std),
            )

        with pyro.plate("data", self.num_observations):
            mix = dist.Categorical(probs=(torch.ones(num_clusters) / num_clusters))
            comp = dist.Independent(dist.Normal(cluster_means, 0.1), 1)
            obs = pyro.sample(
                "obs", dist.MixtureSameFamily(mix, comp), obs=self.observed_data
            )

        return cluster_means, obs


class StochasticInfiniteGMMModel(FiniteGMMModel):

    max_num_clusters = float("inf")
    poisson_rate = 9

    def __init__(
        self,
        data_path=None,
        validation_data_path=None,
        cluster_means_dim=2,
        num_observations=NUM_OBSERVATIONS,
        cluster_means_prior_std=10,
        num_subsample=100,
    ):
        self.observed_data = None
        self.ground_truth = None
        self.validation_data_path = validation_data_path
        self.cluster_means_dim = cluster_means_dim
        self.num_observations = num_observations
        self.cluster_means_prior_std = cluster_means_prior_std
        self.num_subsample = num_subsample

        if not (data_path is None):
            with np.load(data_path) as d:
                self.observed_data = torch.tensor(d["obs"])
                self.ground_truth = {k: d[k] for k in d.files}

                logging.info("Ground truth")
                logging.info(20 * "=" + "\n")
                num_clusters = self.ground_truth["num_clusters"]
                logging.info(f"Number of clusters: {num_clusters}")
                for i in range(num_clusters):
                    m = self.ground_truth[f"mean_{i}"]
                    logging.info(f"Mean {i}: {m}")

    def model(self):
        num_clusters = (
            pyro.sample(
                "num_clusters",
                dist.Poisson(self.poisson_rate),
                infer={"branching": True},
            )
            + 1
        )
        num_clusters = int(num_clusters.item())
        cluster_means = torch.zeros((num_clusters, self.cluster_means_dim))
        for k in range(num_clusters):
            cluster_mean = torch.zeros(self.cluster_means_dim)
            cluster_std = self.cluster_means_prior_std * torch.eye(
                self.cluster_means_dim
            )
            cluster_means[k] = pyro.sample(
                f"mean_{k}",
                dist.MultivariateNormal(cluster_mean, cluster_std),
            )

        with pyro.plate(
            "data", size=self.num_observations, subsample_size=self.num_subsample
        ) as ind:
            mix = dist.Categorical(probs=(torch.ones(num_clusters) / num_clusters))
            comp = dist.Independent(dist.Normal(cluster_means, 0.1), 1)
            obs = pyro.sample(
                "obs",
                dist.MixtureSameFamily(mix, comp),
                obs=self.observed_data.index_select(0, ind),
            )

        return cluster_means, obs

    def evaluation(self, posterior_samples, ground_truth_weights=None):
        # Load validation data.
        with np.load(self.validation_data_path) as d:
            validation_data = torch.tensor(d["obs"])

        log_posterior_densities = torch.zeros((len(posterior_samples)))
        for i, trace in enumerate(posterior_samples):
            predictive_dist = trace.nodes["obs"]["fn"]

            # Need to create new distribution which has the correct batch shape
            num_clusters = int(trace.nodes["num_clusters"]["value"] + 1)
            mix = dist.Categorical(probs=(torch.ones(num_clusters) / num_clusters))
            cluster_means = predictive_dist.component_distribution.mean[0, :, :]
            comp = dist.Independent(dist.Normal(cluster_means, 0.1), 1)
            predictive_dist = dist.MixtureSameFamily(mix, comp)
            predictive_dist = predictive_dist.expand(
                (int(validation_data.size(0) / 2),)
            )

            log_posterior_densities[i] = predictive_dist.log_prob(
                validation_data[: int(validation_data.size(0) / 2)]
            ).sum()

        lppd = torch.mean(log_posterior_densities)
        return lppd
