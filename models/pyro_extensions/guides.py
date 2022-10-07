import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

from torch import nn
from pyro import poutine
from pyro.poutine.util import site_is_subsample
from pyro.infer.autoguide.initialization import init_to_mean
from torch.distributions import biject_to
from pyro.distributions.util import sum_rightmost


class NormalNonReparam(dist.Normal):
    has_rsample = False


class AutoSLPContinuousGuide(nn.Module):
    """Based on https://docs.pyro.ai/en/stable/_modules/pyro/infer/autoguide/guides.html#AutoContinuous."""

    def __init__(self, prototype_trace: poutine.Trace):
        super().__init__()
        self._prototype_trace = prototype_trace

        self._unconstrained_shapes = dict()
        for name, site in self._iter_guide_nodes(self._prototype_trace):
            # Collect the shapes of unconstrained values.
            # These may differ from the shapes of constrained values.

            self._unconstrained_shapes[name] = (
                biject_to(site["fn"].support).inv(site["value"]).shape
            )

        self.latent_dim = sum(
            _product(shape) for shape in self._unconstrained_shapes.values()
        )

    def _iter_guide_nodes(self, prototype_trace: poutine.Trace):
        """Only iterates over the nodes which are relevant to construct the guide."""
        for name, site in prototype_trace.iter_stochastic_nodes():
            if (
                site["type"] != "sample"
                or site["is_observed"]
                or site_is_subsample(site)
            ):
                continue

            yield name, site

    def get_posterior(self):
        raise NotImplementedError()

    def sample_latent(self):
        d = self.get_posterior()
        return pyro.sample("_guide_latent", d, infer={"is_auxiliary": True})

    def _unpack_latent(self, latent: torch.tensor):
        """Convert N-dimensional vector into an iterator of (key, value) pairs."""
        batch_shape = latent.shape[:-1]
        pos = 0
        for name, site in self._iter_guide_nodes(self._prototype_trace):
            constrained_shape = site["value"].shape
            unconstrained_shape = self._unconstrained_shapes[name]
            size = _product(unconstrained_shape)
            event_dim = (
                site["fn"].event_dim + len(unconstrained_shape) - len(constrained_shape)
            )
            unconstrained_shape = torch.broadcast_shapes(
                unconstrained_shape, batch_shape + (1,) * event_dim
            )
            unconstrained_value = latent[..., pos : pos + size].view(
                unconstrained_shape
            )
            yield site, unconstrained_value
            pos += size

    def _pack_latent(self, trace: poutine.Trace):
        """Convert trace from program into N-dimensional vector."""
        batch_shape = None
        for name, site in trace.iter_stochastic_nodes():
            batch_shape = site["value"].shape[:-1]
            break

        post_dist = self.get_posterior()
        latent = torch.zeros(batch_shape + post_dist.event_shape)

        # Loop over prototype trace to ensure same ordering. This might not be
        # necessary but I am doing it just to be safe.
        pos = 0
        for name, site in self._iter_guide_nodes(self._prototype_trace):
            size = _product(self._unconstrained_shapes[name])
            value = biject_to(site["fn"].support).inv(trace.nodes[name]["value"])
            # latent[..., pos : pos + size] = trace.nodes[name]["value"]
            latent[..., pos : pos + size] = value
            pos += size

        return latent

    def forward(self):
        latent = self.sample_latent()

        result = dict()
        for site, unconstrained_value in self._unpack_latent(latent):
            name = site["name"]

            transform = biject_to(site["fn"].support)
            value = transform(unconstrained_value)

            log_density = transform.inv.log_abs_det_jacobian(
                value,
                unconstrained_value,
            )
            log_density = sum_rightmost(
                log_density,
                log_density.dim() - value.dim() + site["fn"].event_dim,
            )
            delta_dist = dist.Delta(
                value, log_density=log_density, event_dim=site["fn"].event_dim
            )

            result[name] = pyro.sample(name, delta_dist)

        return result

    def log_joint_trace(self, trace: poutine.Trace):
        """Evaluate log probability of trace under the posterior."""
        post_dist = self.get_posterior()
        latent = self._pack_latent(trace)
        return post_dist.log_prob(latent)


class AutoSLPNormalGuide(AutoSLPContinuousGuide):
    def __init__(self, prototype_trace: poutine.Trace, init_loc_fn=init_to_mean):
        super().__init__(prototype_trace)
        self._init_loc_fn = init_loc_fn

        self.loc = nn.Parameter(self._init_loc())
        # self.loc = nn.Parameter(torch.zeros((self.latent_dim,)))
        # self.loc = nn.Parameter(dist.Normal(0, 1).sample((self.latent_dim,)))
        self.log_scale = nn.Parameter(
            torch.zeros((self.latent_dim,))  # + torch.log(torch.tensor(0.1))
        )

    def _init_loc(self):
        parts = []
        for _, site in self._iter_guide_nodes(self._prototype_trace):
            # if (
            #     site["type"] != "sample"
            #     or site["is_observed"]
            #     or site_is_subsample(site)
            # ):
            #     continue
            # parts.append(self._init_loc_fn(site).reshape(-1))
            constrained_value = self._init_loc_fn(site).detach()
            unconstrained_value = biject_to(site["fn"].support).inv(constrained_value)
            parts.append(unconstrained_value.reshape(-1))

        latent = torch.cat(parts)
        assert latent.size() == (self.latent_dim,)
        return latent

    def get_posterior(self):
        # We need to .to_event(1) in order for the batching to work properly.
        # This changes the rightmost batch dimension to an event dimension.
        # We need to use the NonResample version of the Normal distribution to
        # avoid using the reparameterized gradient estimator.
        return NormalNonReparam(self.loc, torch.exp(self.log_scale)).to_event(1)


class AutoSLPNormalReparamGuide(AutoSLPNormalGuide):
    def get_posterior(self):
        # We need to .to_event(1) in order for the batching to work properly.
        # This changes the rightmost batch dimension to an event dimension.
        # We need to use the NonResample version of the Normal distribution to
        # avoid using the reparameterized gradient estimator.
        return dist.Normal(self.loc, torch.exp(self.log_scale)).to_event(1)


def _product(array):
    result = 1
    for x in array:
        result *= x
    return result