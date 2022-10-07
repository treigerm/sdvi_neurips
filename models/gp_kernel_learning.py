import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import logging
import pyro.contrib.gp as gp

from torch.distributions import biject_to

from .abstract_model import AbstractModel
from .pyro_extensions.guides import AutoSLPNormalReparamGuide


class GPKernelLearning(AbstractModel):
    does_lppd_evaluation = True
    slps_identified_by_discrete_samples = True

    input_dim = 1

    def __init__(self, data_path, jitter=1e-6):
        self.X, self.y, self.X_val, self.y_val = self.load_data(data_path)

        self.jitter = jitter

    @staticmethod
    def load_data(data_path):
        data = torch.tensor(np.loadtxt(data_path, delimiter=","))
        xs = data[:, 0]
        ys = data[:, 1]
        xs -= xs.min()
        xs /= xs.max()
        ys -= ys.mean()
        ys *= 4 / (ys.max() - ys.min())

        # Keep 10 % of data for validation.
        val_ix = round(xs.size(0) * 0.9)
        xs, xs_val = xs[:val_ix], xs[val_ix:]
        ys, ys_val = ys[:val_ix], ys[val_ix:]

        return xs, ys, xs_val, ys_val

    def sample_kernel_fn(self, address_prefix: str) -> gp.kernels.Kernel:
        kernel_type = pyro.sample(
            f"{address_prefix}kernel_type",
            dist.Categorical(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])),
            infer={"branching": True},
        )

        if kernel_type == 0.0:
            # Rational Quadratic kernel
            rq_kernel = gp.kernels.RationalQuadratic(
                input_dim=self.input_dim,
            )
            rq_kernel.lengthscale = pyro.nn.PyroSample(dist.InverseGamma(2.0, 1.0))
            rq_kernel.scale_mixture = pyro.nn.PyroSample(dist.InverseGamma(2.0, 1.0))
            return rq_kernel
        elif kernel_type == 1.0:
            # Linear kernel
            linear_kernel = gp.kernels.Polynomial(input_dim=self.input_dim, degree=1)
            linear_kernel.bias = pyro.nn.PyroSample(dist.InverseGamma(2.0, 1.0))
            return linear_kernel
        elif kernel_type == 2.0:
            # Squared Exponential kernel
            rbf_kernel = gp.kernels.RBF(
                input_dim=self.input_dim,
            )
            rbf_kernel.lengthscale = pyro.nn.PyroSample(dist.InverseGamma(2.0, 1.0))
            return rbf_kernel
        elif kernel_type == 3.0:
            # Periodic
            periodic_kernel = gp.kernels.Periodic(
                input_dim=self.input_dim,
                variance=torch.tensor(1.0),
            )
            periodic_kernel.lengthscale = pyro.nn.PyroSample(
                dist.InverseGamma(2.0, 1.0)
            )
            periodic_kernel.period = pyro.nn.PyroSample(dist.InverseGamma(2.0, 1.0))
            return periodic_kernel
        elif kernel_type == 4.0:
            # Sum
            left_child = self.sample_kernel_fn(f"{address_prefix}sum_left_")
            right_child = self.sample_kernel_fn(f"{address_prefix}sum_right_")
            return gp.kernels.Sum(left_child, right_child)
        elif kernel_type == 5.0:
            # Product
            left_child = self.sample_kernel_fn(f"{address_prefix}times_left_")
            right_child = self.sample_kernel_fn(f"{address_prefix}times_right_")
            return gp.kernels.Product(left_child, right_child)
        else:
            raise ValueError(f"Unkown kernel type: {kernel_type}")

    def __call__(self):
        # Sample kernel function
        kernel_fn = self.sample_kernel_fn("")

        std = pyro.sample("std", dist.HalfNormal(1))

        # Create covariance matrix
        N = self.X.size(0)
        Kff = kernel_fn(self.X)
        # The constant kernel has problems because it uses .expand() internally.
        # To avoid the problem we need to clone the covariance matrix.
        Kff = Kff.type(self.X.dtype).clone()
        Kff.view(-1)[:: N + 1] += self.jitter + torch.pow(
            std, 2
        )  # add noise to diagonal
        Lff = torch.linalg.cholesky(Kff)

        zero_loc = self.X.new_zeros(N)
        pyro.sample("y", dist.MultivariateNormal(zero_loc, scale_tril=Lff), obs=self.y)

        # Sample data from standard normal
        return kernel_fn

    def make_parameter_plots(self, results, guide, branching_trace, file_prefix):
        if isinstance(guide, AutoSLPNormalReparamGuide):
            means = results["loc"]
            scale = [np.exp(v) for v in results["log_scale"]]
        else:
            logging.info(f"Parameter plotting for guide {guide} not supported.")
            return

        means = [
            [(site, v) for site, v in guide._unpack_latent(torch.tensor(cm))]
            for cm in means
        ]
        scale = [
            [(site, v) for site, v in guide._unpack_latent(torch.tensor(cs))]
            for cs in scale
        ]
        num_params = len(means[-1])

        # Plot final distributions
        fig, axs = plt.subplots(num_params, 1, figsize=(10, 4 * num_params))
        for ix in range(num_params):
            site = means[-1][ix][0]
            transform = biject_to(site["fn"].support)

            mean, std = means[-1][ix][1], scale[-1][ix][1]
            q_dist = dist.Normal(mean, std)
            xs = torch.linspace(mean - 3 * std, mean + 3 * std, 100)
            constrained_xs = transform(xs)
            log_densities = q_dist.log_prob(xs) + transform.inv.log_abs_det_jacobian(
                constrained_xs, xs
            )
            axs[ix].plot(constrained_xs, log_densities.exp())
            axs[ix].set_title(site["name"])

        fig.tight_layout()
        fig.savefig(f"{file_prefix}_final_marginals.jpg")

        # Plot evolution of the means
        fig, axs = plt.subplots(num_params, 1, figsize=(10, 4 * num_params))
        for ix in range(num_params):
            site = means[0][ix][0]
            transform = biject_to(site["fn"].support)

            param_means = torch.tensor([x[ix][1] for x in means])
            constrained_param_means = transform(param_means)
            axs[ix].plot(constrained_param_means)
            axs[ix].set_title(f"{site['name']} mean")
            axs[ix].set_xlabel("Iteration")
            axs[ix].set_ylabel("Value")

        fig.tight_layout()
        fig.savefig(f"{file_prefix}_params.jpg")
        plt.close("all")

    def evaluation(self, posterior_samples, ground_truth_weights=None):
        post_kernels = self.extract_posterior_kernels(posterior_samples)
        noises = [trace.nodes["std"]["value"] for trace in posterior_samples]

        log_p = torch.tensor(0.0)
        for kernel_fn, noise in zip(post_kernels, noises):
            gp_mean, gp_cov = self.gp_analytic_posterior(
                kernel_fn, self.X, self.X_val, self.y, noise, self.jitter, full_cov=True
            )
            log_p += (
                dist.MultivariateNormal(gp_mean, gp_cov).log_prob(self.y_val).detach()
            )

        return log_p / len(posterior_samples)

    def plot_posterior_samples(self, posterior_samples, fname):
        post_kernels = self.extract_posterior_kernels(posterior_samples)
        noises = [trace.nodes["std"]["value"] for trace in posterior_samples]

        new_xs = torch.linspace(0, 1, 500)
        posterior_fs = torch.zeros((len(post_kernels), new_xs.size(0)))
        gp_means = torch.zeros((len(post_kernels), new_xs.size(0)))
        gp_vars = torch.zeros((len(post_kernels), new_xs.size(0)))
        for ix in range(len(post_kernels)):
            with torch.no_grad():
                gp_post_mean, gp_post_cov = self.gp_analytic_posterior(
                    post_kernels[ix],
                    self.X,
                    new_xs,
                    self.y,
                    noises[ix],
                    self.jitter,
                    full_cov=True,
                )
            posterior_fs[ix, :] = (
                dist.MultivariateNormal(gp_post_mean, gp_post_cov).sample().detach()
            )

        f_post_mean = posterior_fs.mean(dim=0)
        f_post_std = posterior_fs.std(dim=0)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(new_xs, f_post_mean, label="Post mean", color="red")
        ax.fill_between(
            new_xs,
            f_post_mean - 2 * f_post_std,
            f_post_mean + 2 * f_post_std,
            color="red",
            alpha=0.2,
        )
        num_samples_to_plot = min(5, len(post_kernels))
        for ix in range(num_samples_to_plot):
            ax.plot(new_xs, posterior_fs[ix, :], color="green", alpha=0.3)

        ax.scatter(self.X, self.y, label="Data", color="black")
        ax.scatter(self.X_val, self.y_val, label="Held-out data")
        ax.set_xlim((-0.01, 1.01))
        ax.legend(loc="upper left")
        fig.savefig(fname)
        plt.close("all")

    @staticmethod
    def extract_posterior_kernels(posterior_samples):
        post_kernels = [trace.nodes["_RETURN"]["value"] for trace in posterior_samples]
        for ix in range(len(post_kernels)):
            for name, s in posterior_samples[ix].iter_stochastic_nodes():
                if name in ["std", "y"] or "kernel_type" in name:
                    continue

                if isinstance(post_kernels[ix], gp.kernels.Sum) or isinstance(
                    post_kernels[ix], gp.kernels.Product
                ):
                    names = name.split(".")
                    kern_mod = post_kernels[ix]._modules[names[0]]
                    for jx in range(len(names) - 2):
                        kern_mod = kern_mod._modules[names[jx + 1]]
                    setattr(kern_mod, names[-1], s["value"])
                else:
                    setattr(post_kernels[ix], name, s["value"])
        return post_kernels

    @staticmethod
    def gp_analytic_posterior(
        kernel_fn: gp.kernels.Kernel,
        X: torch.tensor,
        new_xs: torch.tensor,
        y: torch.tensor,
        noise: torch.tensor,
        jitter: float,
        full_cov: bool = False,
    ):
        N = X.size(0)
        Kff = kernel_fn(X).contiguous()
        Kff = Kff.type(X.dtype).clone()
        Kff.view(-1)[:: N + 1] += jitter + torch.pow(noise, 2)
        Lff = torch.linalg.cholesky(Kff)

        gp_post_mean, gp_post_cov = gp.util.conditional(
            new_xs, X, kernel_fn, y, Lff=Lff, jitter=jitter, full_cov=full_cov
        )
        if full_cov:
            M = new_xs.size(0)
            gp_post_cov = gp_post_cov.contiguous()
            gp_post_cov.view(-1, M * M)[:, :: M + 1] += torch.pow(noise, 2)
        else:
            gp_post_cov = gp_post_cov + torch.pow(noise, 2)
        return gp_post_mean, gp_post_cov