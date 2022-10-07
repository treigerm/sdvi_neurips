import os
import pickle
import logging
import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.poutine.util import site_is_subsample
import copy
import tqdm
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Callable

from .handlers import BranchingTraceMessenger
from .resource_allocation import AbstractUtility
from .util import get_sample_addresses


@dataclass
class SLPInfo:
    initial_trace: poutine.Trace
    num_proposed: int = 0
    num_selected: int = 0
    log_marginal_likelihood: torch.Tensor = None
    mcmc_samples: list[list[poutine.Trace]] = None
    log_weights: torch.Tensor = None
    branching_sample_values: OrderedDict[str, torch.Tensor] = field(
        default_factory=OrderedDict
    )


@dataclass
class IterationInfo:
    selected_addresses: list[str] = field(default_factory=list)
    log_marginal_likelihoods_per_iteration: dict[str, list[torch.Tensor]] = field(
        default_factory=dict
    )


class DCC:
    def __init__(
        self,
        model,
        utility: AbstractUtility,
        num_iterations: int,
        num_chains: int,
        num_init_mcmc: int,
        num_mcmc_steps_per_iteration: int,
        num_pi_mais_samples: int,
        num_slp_samples: int,
        min_num_proposed: int,
        sigma: float = 1.0,
        mcmc_sample_hide_fn: Callable = lambda x: False,
        checkpoint_dir: str = None,
    ):
        self.model = model
        self.utility = utility

        self.num_iterations = num_iterations
        self.num_slp_samples = num_slp_samples
        self.min_num_proposed = min_num_proposed

        self.num_chains = num_chains
        self.num_init_mcmc = num_init_mcmc
        self.num_mcmc_steps_per_iteration = num_mcmc_steps_per_iteration

        self.num_pi_mais_samples = num_pi_mais_samples

        self.local_mcmc = LocalMCMC(sigma, mcmc_sample_hide_fn)

        self.checkpoint_dir = checkpoint_dir

    def find_slps(self, *args, **kwargs) -> tuple[set[str], dict[str, SLPInfo]]:
        slp_traces = set()
        slp_info: dict[str, SLPInfo] = dict()
        for _ in range(self.num_slp_samples):
            # trace = poutine.trace(self.model).get_trace(*args, **kwargs)
            with torch.no_grad():
                with pyro.poutine.trace_messenger.TraceMessenger() as tmsngr:
                    with BranchingTraceMessenger() as btmsngr:
                        ret = pyro.poutine.block(self.model, hide_types=["param"])()

            trace = tmsngr.get_trace()
            # Need to manually add the return node if we use trace messenger as a 
            # context.
            trace.add_node("_RETURN", name="_RETURN", type="return", value=ret)

            addresses = get_sample_addresses(trace)
            address_trace = ",".join(addresses)
            if address_trace in slp_traces:
                slp_info[address_trace].num_proposed += 1
            else:
                slp_traces.add(address_trace)
                slp_info[address_trace] = SLPInfo(
                    trace, branching_sample_values=btmsngr.get_sampled_values()
                )

        return slp_traces, slp_info

    def run(
        self, *args, **kwargs
    ) -> tuple[dict[str, SLPInfo], IterationInfo]:
        # Find SLPs
        A_total, slps_info = self.find_slps(*args, **kwargs)
        A_active: set[str] = set()
        initialized_slps: set[str] = set()
        iteration_info = IterationInfo()

        max_log_weight = torch.tensor(float("-inf"))

        for iteration_ix in tqdm.tqdm(range(self.num_iterations), desc="Iterations"):
            # Add models into active set
            for addr_trace, slp_info in slps_info.items():
                if slp_info.num_proposed >= self.min_num_proposed:
                    A_active.add(addr_trace)

            # For all new models initialise MCMC chains
            slps_to_intialize = (
                A_active - initialized_slps
            )  # Here minus is the set difference.
            for addr_trace in tqdm.tqdm(slps_to_intialize):
                # Run N MCMC chains for N_init iterations
                (
                    slps_info[addr_trace].mcmc_samples,
                    max_lw,
                ) = self.local_mcmc.run_burn_in(
                    self.model,
                    addr_trace,
                    slps_info[addr_trace].branching_sample_values,
                    self.num_chains,
                    self.num_init_mcmc,
                    slps_info[addr_trace].initial_trace,
                )
                max_log_weight = torch.max(max_log_weight, max_lw)

                # Use PI MAIS to estimate marginal likelihood
                log_Z, lws = run_pi_mais(
                    self.model,
                    addr_trace,
                    slps_info[addr_trace].branching_sample_values,
                    slps_info[addr_trace].mcmc_samples,
                    self.num_pi_mais_samples,
                )
                slps_info[addr_trace].log_marginal_likelihood = log_Z
                slps_info[addr_trace].log_weights = lws
                slps_info[addr_trace].num_selected += 1

                initialized_slps.add(addr_trace)
                iteration_info.log_marginal_likelihoods_per_iteration[addr_trace] = [log_Z]

            # Choose model based on utility
            addr_trace2num_selected = {
                at: slps_info[at].num_selected for at in A_active
            }
            addr_trace2log_weights = {at: slps_info[at].log_weights for at in A_active}
            utilities = self.utility.utility_function(
                addr_trace2log_weights, addr_trace2num_selected, max_log_weight
            )
            selected_addr_trace = max(utilities, key=utilities.get)

            # Continue MCMC chains
            last_traces = [
                chain[-1] for chain in slps_info[selected_addr_trace].mcmc_samples
            ]
            mcmc_samples, max_lw = self.local_mcmc.run_mcmc(
                self.model,
                selected_addr_trace,
                slps_info[selected_addr_trace].branching_sample_values,
                self.num_chains,
                self.num_mcmc_steps_per_iteration,
                last_traces,
            )
            max_log_weight = torch.max(max_log_weight, max_lw)
            log_Z, lws = run_pi_mais(
                self.model,
                selected_addr_trace,
                slps_info[selected_addr_trace].branching_sample_values,
                mcmc_samples,
                self.num_pi_mais_samples,
            )

            # Update MCMC samples, marginal likelihood and log weights
            slps_info, iteration_info = self.update_data_structures(
                slps_info, iteration_info, selected_addr_trace, mcmc_samples, log_Z, lws
            )
            if not (self.checkpoint_dir is None) and ((iteration_ix+1) % 50 == 0):
                fname = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration_ix+1}.pickle")
                logging.info(f"Saving checkpoint to {fname}")
                with open(fname, "wb") as f:
                    pickle.dump(
                        {"slps_info": slps_info, "iteration_info": iteration_info, "iteration_ix": iteration_ix}, 
                        f
                    )

        # Combine samples into overall posterior using marginal likelihood estimates
        return slps_info, iteration_info

    @staticmethod
    def update_data_structures(
        slps_info: dict[str, SLPInfo],
        iteration_info: IterationInfo,
        selected_addr_trace: str,
        mcmc_samples: list[list[poutine.Trace]],
        log_marginal_likelihood: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> tuple[dict[str, SLPInfo], IterationInfo]:
        # Add new MCMC samples to SLP
        for ix, chain in enumerate(mcmc_samples):
            slps_info[selected_addr_trace].mcmc_samples[ix] = (
                slps_info[selected_addr_trace].mcmc_samples[ix] + chain
            )
        # Update marginal likelihood estimate
        slps_info[selected_addr_trace].log_marginal_likelihood = 0.5 * (
            log_marginal_likelihood + slps_info[selected_addr_trace].log_marginal_likelihood
        )
        slps_info[selected_addr_trace].log_marginal_likelihood = torch.logsumexp(
            torch.tensor([log_marginal_likelihood, slps_info[selected_addr_trace].log_marginal_likelihood]),
            dim=0
        ) - torch.log(torch.tensor(2.0))
        # Add new log weights to SLP
        slps_info[selected_addr_trace].log_weights = torch.cat(
            (slps_info[selected_addr_trace].log_weights, log_weights)
        )
        slps_info[selected_addr_trace].num_selected += 1

        # Update marginal likelihood per iteration for selected SLP and all others
        iteration_info.log_marginal_likelihoods_per_iteration[selected_addr_trace].append(
            slps_info[selected_addr_trace].log_marginal_likelihood
        )
        for addr_trace in iteration_info.log_marginal_likelihoods_per_iteration.keys():
            if addr_trace == selected_addr_trace:
                continue
            previous_log_marginal_likelihood = (
                iteration_info.log_marginal_likelihoods_per_iteration[addr_trace][-1]
            )
            iteration_info.log_marginal_likelihoods_per_iteration[addr_trace].append(
                previous_log_marginal_likelihood
            )

        iteration_info.selected_addresses.append(selected_addr_trace)

        return slps_info, iteration_info


def run_pi_mais(
    model,
    addr_trace: str,
    branching_sample_values: OrderedDict[str, torch.Tensor],
    mcmc_samples: list[list[poutine.Trace]],
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cond_model = poutine.condition(model, data=branching_sample_values)

    num_chains = len(mcmc_samples)
    num_samples_per_chain = len(mcmc_samples[0])
    log_weights = torch.zeros((num_chains, num_samples_per_chain, num_samples))
    for i, chain in enumerate(mcmc_samples):
        for j, sample in enumerate(chain):
            # Extract proposal distribution
            proposal_dist = PIMAISProposal(sample, exclude_addresses=set(branching_sample_values.keys()))
            for k in range(num_samples):
                # Sample from proposal
                q_trace, q_trace_lw = proposal_dist.sample_trace()

                # Evaluate log weight
                try:
                    model_trace = poutine.trace(
                        poutine.replay(cond_model, trace=q_trace)
                    ).get_trace()
                except:
                    log_weights[i, j, k] = torch.tensor(float("-inf"))
                    continue
                addresses = get_sample_addresses(
                    model_trace, included_addresses=set(branching_sample_values.keys())
                )

                model_trace_in_slp = ",".join(addresses) == addr_trace
                if model_trace_in_slp:
                    lw = model_trace.log_prob_sum() - q_trace_lw
                    log_weights[i, j, k] = lw
                else:
                    log_weights[i, j, k] = torch.tensor(float("-inf"))

    log_Z = torch.logsumexp(log_weights, dim=(0, 1, 2)) - torch.log(
        torch.tensor(num_chains * num_samples_per_chain * num_samples)
    )
    return log_Z, torch.reshape(log_weights, (-1,))


class PIMAISProposal:
    def __init__(self, trace: poutine.Trace, exclude_addresses: set[str] = set()):
        self.trace = trace
        self.exclude_addresses = exclude_addresses

    def sample_trace(self) -> tuple[poutine.Trace, torch.Tensor]:
        new_trace = copy.deepcopy(self.trace)
        log_weight = torch.tensor(0.0)

        for name, site in self.trace.iter_stochastic_nodes():
            if (
                site["type"] != "sample"
                or site["is_observed"]
                or site_is_subsample(site)
                or name in self.exclude_addresses
            ):
                continue

            if len(site["fn"].event_shape) > 0:
                q_dist = dist.MultivariateNormal(
                    site["value"], torch.eye(site["value"].shape[0])
                )
            elif site["fn"].support == dist.constraints.positive:
                q_dist = dist.LogNormal(site["value"], 1.0)
            else:
                q_dist = dist.Normal(site["value"], 1.0)
            q_sample = q_dist.sample()

            log_weight += q_dist.log_prob(q_sample)
            new_trace.nodes[name]["value"] = q_sample

        return new_trace, log_weight


class LocalMCMC:
    def __init__(self, sigma: float, hide_fn: Callable):
        # Parameter indicating the std of local proposals
        self.sigma = sigma
        self.hide_fn = hide_fn

    def run_burn_in(
        self,
        model,
        address_trace: str,
        branching_sample_values: OrderedDict[str, torch.Tensor],
        num_chains: int,
        num_steps: int,
        initial_trace: poutine.Trace,
    ) -> tuple[list[list[poutine.Trace]], torch.Tensor]:
        initial_traces = num_chains * [initial_trace]
        return self.run_mcmc(
            model,
            address_trace,
            branching_sample_values,
            num_chains,
            num_steps,
            initial_traces,
        )

    def run_mcmc(
        self,
        model,
        address_trace: str,
        branching_sample_values: OrderedDict[str, torch.Tensor],
        num_chains: int,
        num_steps: int,
        initial_traces: list[poutine.Trace],
    ) -> tuple[list[list[poutine.Trace]], torch.Tensor]:
        chains: list[list[poutine.Trace]] = []
        max_log_weight = torch.tensor(float("-inf"))

        cond_model = poutine.condition(model, data=branching_sample_values)

        for init_trace in initial_traces:
            sample_addresses = get_sample_addresses(init_trace)
            sample_addresses = [
                addr
                for addr in sample_addresses
                if not (addr in branching_sample_values.keys())
            ]
            chain, max_lw = self.run_single_chain(
                cond_model,
                ",".join(sample_addresses),
                sample_addresses,
                branching_sample_values,
                num_steps,
                init_trace,
            )
            chains.append(chain)
            max_log_weight = torch.max(max_log_weight, max_lw)

        return chains, max_lw

    def run_single_chain(
        self,
        model,
        address_trace: str,
        sample_addresses: list[str],
        branching_sample_values: OrderedDict[str, torch.Tensor],
        num_steps: int,
        initial_trace: poutine.Trace,
    ) -> tuple[list[poutine.Trace], torch.Tensor]:
        trace = copy.deepcopy(initial_trace)
        # print(trace.nodes.keys())

        max_log_weight = torch.tensor(float("-inf"))
        samples = []
        for _ in range(num_steps):
            # Sample address for which to do new sample
            # sample_addresses = get_sample_addresses(trace)
            update_ix = dist.Categorical(
                probs=torch.ones(len(sample_addresses)) / len(sample_addresses)
            ).sample()
            update_address = sample_addresses[update_ix]

            # Resample from prior at given address
            new_trace = copy.deepcopy(trace)
            # new_trace.nodes[update_address]["value"] = new_trace.nodes[update_address][
            #     "fn"
            # ].sample()
            new_value, log_proposal_ratio = self.local_proposal(
                new_trace.nodes[update_address]["fn"],
                trace.nodes[update_address]["value"],
            )
            new_trace.nodes[update_address]["value"] = new_value

            # Check if new trace is valid
            try:
                with torch.no_grad():
                    replayed_trace = poutine.trace(
                        poutine.replay(model, trace=new_trace)
                    ).get_trace()
                new_addresses = get_sample_addresses(replayed_trace)
                # New trace is valid if and only if it falls within the same SLP
                new_trace_is_valid = ",".join(new_addresses) == address_trace
            except:
                new_trace_is_valid = False

            if new_trace_is_valid:
                # Calculate accept-reject step of new model.
                log_acceptance_ratio = (
                    replayed_trace.log_prob_sum() - trace.log_prob_sum()
                )

                # Ratio of proposal distributions, in this case the prior
                # prior_new = replayed_trace.nodes[update_address]["fn"]
                # prior_old = trace.nodes[update_address]["fn"]
                # log_acceptance_ratio += prior_new.log_prob(
                #     trace.nodes[update_address]["value"]
                # ) - prior_old.log_prob(replayed_trace.nodes[update_address]["value"])
                log_acceptance_ratio += log_proposal_ratio

                log_acceptance_ratio = torch.min(torch.tensor(0), log_acceptance_ratio)

                u = dist.Uniform(0.0, 1.0).sample()
                if u < log_acceptance_ratio.exp():
                    for name in branching_sample_values.keys():
                        # This is necessary so that the resulting traces can be replayed later.
                        replayed_trace.nodes[name]["is_observed"] = False
                    trace = copy.deepcopy(replayed_trace)
            else:
                # Simply reject new trace if it falls outside the SLP.
                trace = copy.deepcopy(trace)

            log_weight = trace.log_prob_sum()
            max_log_weight = torch.max(max_log_weight, log_weight)

            samples.append(self.filter_trace(trace))

        return samples, max_log_weight

    def local_proposal(
        self, distribution: dist.Distribution, current_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not distribution.support.is_discrete:
            if distribution.support == dist.constraints.positive:
                proposal = dist.LogNormal(current_value, self.sigma)
                new_value = proposal.sample()
                log_proposal_ratio = dist.LogNormal(new_value, self.sigma).log_prob(current_value) - proposal.log_prob(new_value)
            else:
                proposal = dist.Normal(current_value, self.sigma)
                new_value = proposal.sample()
                log_proposal_ratio = torch.tensor(
                    0.0
                )  # Is 0 because the local gaussian proposal is symmetric
        else:
            raise ValueError("Discrete distribution not supported yet.")
        return new_value, log_proposal_ratio

    def filter_trace(self, trace: poutine.Trace):
        nodes_to_delete = []
        for name, msg in trace.nodes.items():
            if self.hide_fn(msg):
                nodes_to_delete.append(name)

        for name in nodes_to_delete:
            trace.remove_node(name)

        return trace