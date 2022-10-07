import torch
import pyro
import pyro.distributions as dist
import tqdm
import copy
import logging

from pyro import poutine
from pyro.infer.autoguide.initialization import init_to_mean
from torch import multiprocessing
from collections import OrderedDict

from .handlers import (
    BranchingTraceMessenger,
    LogJointBranchingTraceHandler,
)
from .guides import (
    AutoSLPNormalGuide,
    AutoSLPNormalReparamGuide,
)
from .resource_allocation import (
    SuccessiveHalving,
    RepeatedSuccessiveHalving,
)
from .util import get_sample_addresses

multiprocessing.set_sharing_strategy("file_system")


def slp_identified_by_sampled_values(
    branching_trace: str, branching_sampled_values: OrderedDict
):
    string_represention = "".join(
        [str(v.item()) for v in branching_sampled_values.values()]
    )
    return branching_trace == string_represention


class SDVI:
    def __init__(
        self,
        model,
        learning_rate,
        guide_class_name,
        utility_class,
        slps_identified_by_discrete_samples=False,
        find_slp_samples=100,
        forward_kl_iter=100,
        forward_kl_num_particles=1,
        initial_num_iterations=100,
        num_iterations_per_step=100,
        num_steps=100,
        exclusive_kl_num_particles=1,
        elbo_estimate_num_particles=100,
        iwae_num_inner=None,
        autoguide_hide_vars=[],
        num_parallel_processes=10,
        init_loc_fn=init_to_mean,  # Initialization of the means of the AutoGuides
        use_iwae_for_weights=False,
        save_metrics_every_n=100,
    ):
        self.num_parallel_processes = num_parallel_processes

        self.model = model
        self.learning_rate = learning_rate

        self.slps_identified_by_discrete_samples = slps_identified_by_discrete_samples
        self.find_slp_samples = find_slp_samples
        self.smoothing_epsilon = None

        self.use_iwae_for_weights = use_iwae_for_weights

        # Configuration for forward KL phase
        self.forward_kl_iter = forward_kl_iter
        self.forward_kl_num_particles = forward_kl_num_particles

        # Configuration for exclusive KL phase
        self.initial_num_iterations = initial_num_iterations
        self.num_iterations_per_step = num_iterations_per_step
        self.num_steps = num_steps
        self.exclusive_kl_num_particles = exclusive_kl_num_particles
        self.elbo_estimate_num_particles = elbo_estimate_num_particles
        self.iwae_num_inner = iwae_num_inner

        # The variable names from the model that should not be included when creating
        # the guide function. This is mainly useful for variables which are marginalized out.
        self.autoguide_hide_vars = autoguide_hide_vars
        self.init_loc_fn = init_loc_fn

        self.utility_class = utility_class

        if guide_class_name in ["MeanFieldNormal"]:
            self.guide_class_name = guide_class_name
        else:
            raise ValueError(f"Unknown guide class name: {guide_class_name}")

        self.exclusive_kl_results = None
        self.bt2weight = None

        self.save_metrics_every_n = save_metrics_every_n

    def find_slps(self, num_samples, max_slps=1000):
        self.branching_traces = set()
        # Mapping branching_trace => OrderedDict where the dict can be used to condition
        # the model so that it lies in the given SLP.
        self.branching_sample_values = dict()
        self.prototype_traces = dict()

        min_logjoint = torch.tensor(float("inf"))
        for _ in range(num_samples):
            with pyro.poutine.trace_messenger.TraceMessenger() as tmsngr:
                with BranchingTraceMessenger() as btmsngr:
                    pyro.poutine.block(self.model, hide_types=["param"])()

            # bt = btmsngr.get_trace()
            trace = tmsngr.get_trace()
            if self.slps_identified_by_discrete_samples:
                bt = btmsngr.get_trace()
            else:
                addresses = get_sample_addresses(trace)
                bt = ",".join(addresses)

            # Keep track of smallest logjoint density value encountered.
            with torch.no_grad():
                if trace.log_prob_sum() < min_logjoint:
                    min_logjoint = trace.log_prob_sum().detach()

            if not (bt in self.branching_traces):
                self.branching_traces.add(bt)
                self.branching_sample_values[bt] = btmsngr.get_sampled_values()

                # For the prototype trace we only want to keep variables for which
                # we want to create a guide and therefore we might wish to delete
                # some nodes from the trace.
                nodes_to_remove = []
                for name, site in trace.nodes.items():
                    if site["infer"].get("branching", False) or site["is_observed"]:
                        # Since we condition on the branching nodes we do not need
                        # them in our guide.
                        nodes_to_remove.append(name)
                nodes_to_remove += self.autoguide_hide_vars
                for node_name in set(nodes_to_remove):
                    trace.remove_node(node_name)

                self.prototype_traces[bt] = trace

            if len(self.branching_traces) >= max_slps:
                break

        self.smoothing_epsilon = torch.log(torch.tensor(0.01)) + min_logjoint

    def initialise_guides(self):
        self.guides = dict()
        for branching_trace, protype_trace in self.prototype_traces.items():
            if self.slps_identified_by_discrete_samples:
                # The reparameterized gradient estimator is applicable
                if self.guide_class_name == "MeanFieldNormal":
                    self.guides[branching_trace] = AutoSLPNormalReparamGuide(
                        protype_trace, init_loc_fn=self.init_loc_fn
                    )
                else:
                    raise ValueError("Guide class unsupported.")
            else:
                if self.guide_class_name == "MeanFieldNormal":
                    self.guides[branching_trace] = AutoSLPNormalGuide(
                        protype_trace, init_loc_fn=self.init_loc_fn
                    )
                else:
                    raise ValueError("Guide class unsupported.")

    def minimise_forward_kl(self, num_iterations, num_particles=1):
        metrics = dict()

        for branching_trace, guide in self.guides.items():
            # if slp_identified_by_sampled_values(
            #     branching_trace, self.branching_sample_values[branching_trace]
            # ):
            if self.slps_identified_by_discrete_samples:
                continue

            params = {name: [p.tolist()] for name, p in guide.named_parameters()}
            # Generate dataset for density estimation.
            slp_traces = []
            with torch.no_grad():
                for _ in range(num_particles):
                    conditioned_model = poutine.condition(
                        self.model,
                        data=self.branching_sample_values[branching_trace],
                    )
                    trace = poutine.trace(conditioned_model).get_trace()
                    is_in_slp = ~torch.isinf(
                        LogJointBranchingTraceHandler(
                            poutine.replay(conditioned_model, trace=trace),
                            branching_trace,
                            epsilon=torch.tensor(float("-inf")),
                        ).log_prob()
                    )
                    if is_in_slp:
                        slp_traces.append(trace)

            if len(slp_traces) == 0:
                raise ValueError(f"Sampled no valid traces for slp {branching_trace}")

            losses = []

            optimizer = torch.optim.Adam(guide.parameters(), lr=self.learning_rate)
            for _ in range(num_iterations):
                optimizer.zero_grad()

                # Sample from prior.
                loss = torch.tensor(0.0)
                for trace in slp_traces:

                    log_q = guide.log_joint_trace(trace)

                    loss += -log_q / num_particles

                losses.append(loss.detach())
                if loss == 0.0:
                    # Do not do optimization step if we have not sampled any values inside SLP.
                    continue

                loss.backward()
                optimizer.step()
                for n, p in guide.named_parameters():
                    params[n].append(p.tolist())

            logging.info(f"Branching trace: {branching_trace}")
            for n, v in guide.named_parameters():
                if v.numel() <= 10:
                    # Only print parameters if there are less than 10 values to avoid clutter.
                    logging.info(f"{n}: {v.detach()}")
            logging.info("")

            metrics[branching_trace] = {"losses": losses} | params

        return metrics

    def minimise_exclusive_kl_successive_halving(
        self,
        initial_num_iterations,
        num_iterations_per_step,
        num_steps,
        num_particles,
        elbo_estimate_num_particles,
        iwae_num_inner,
    ):
        base_seed = torch.randint(high=int(1e9), size=(1,))[0].item()


        guides_items = list(self.guides.items())
        branching_traces = [k for k, v in guides_items]
        bt2num_selected = {bt: 0 for bt in branching_traces}
        bt2optim_state = {bt: None for bt in branching_traces}
        self.exclusive_kl_results = dict()

        # Initialize set of active models
        active_slps = copy.deepcopy(branching_traces)

        # Calculate number of phases
        num_slps = len(branching_traces)
        num_phases = self.utility_class.calculate_num_phases(num_slps)

        logging.info(f"Split training into {num_phases} phases.")
        for i in range(num_phases):
            # Calculate number of time steps in this phase
            num_optimization_steps = (
                self.utility_class.calculate_num_optimization_steps(len(active_slps))
            )

            # Run optimization on each SLP
            args = [
                (
                    ix,
                    self.prototype_traces[selected_bt],
                    selected_bt,
                    self.branching_sample_values[selected_bt],
                    self.slps_identified_by_discrete_samples,
                    self.guides[selected_bt],
                    self.model,
                    self.learning_rate,
                    bt2optim_state[selected_bt],
                    num_optimization_steps,
                    num_particles,
                    self.smoothing_epsilon,
                    self.estimate_local_elbo,
                    elbo_estimate_num_particles,
                    iwae_num_inner,
                    base_seed,
                    self.save_metrics_every_n,
                )
                for ix, selected_bt in enumerate(active_slps)
            ]
            if self.num_parallel_processes > 1:
                with multiprocessing.get_context("spawn").Pool(
                    self.num_parallel_processes
                ) as p:
                    results = list(  # Turn iterator into list.
                        tqdm.tqdm(  # Progress bar.
                            p.imap(  # Use imap to return results as soon as they are finished
                                inner_loop_star,
                                args,
                            ),
                            total=len(
                                args
                            ),  # Pass in total length for proper progress bar.
                        )
                    )
            else:
                results = list(  # Turn iterator into list.
                    tqdm.tqdm(
                        map(
                            inner_loop_star,
                            args,
                        ),
                        total=len(
                            args
                        ),  # Pass in total length for proper progress bar.
                    )
                )

            # Update metrics
            for ix, selected_bt in enumerate(active_slps):
                metrics, _, optim_state = results[ix]
                bt2optim_state[selected_bt] = optim_state
                bt2num_selected[selected_bt] += 1

                # Append metrics
                if selected_bt in self.exclusive_kl_results:
                    self.exclusive_kl_results[selected_bt] = {
                        k: v + metrics[k]
                        for k, v in self.exclusive_kl_results[selected_bt].items()
                    }
                else:
                    self.exclusive_kl_results[selected_bt] = metrics

            # For every SLP that was not selected all the metrics stay fixed. To keep
            # all the metric arrays the same length we repeat the last value for each
            # metric for as many iterations as we run inference on the single SLP.
            num_metrics_saved = len(metrics["losses"])
            for other_bt in self.branching_traces:
                if other_bt in active_slps:
                    continue

                self.exclusive_kl_results[other_bt] = {
                    k: v + num_metrics_saved * [v[-1]]
                    for k, v in self.exclusive_kl_results[other_bt].items()
                }

            # Remove bottom half of active models
            bt2elbo = {
                bt: self.exclusive_kl_results[bt]["true_elbos"][-1].item()
                for bt in active_slps
            }
            active_slps = self.utility_class.select_active_slps(bt2elbo)

        # Final estimation of the ELBO
        for bt in branching_traces:
            # Get model,guide,branching_trace,branching_sampled_values,
            guide = self.guides[bt]
            elbo, _, _ = estimate_local_elbo_and_iwae(
                self.model,
                guide,
                bt,
                self.branching_sample_values[bt],
                num_monte_carlo=self.elbo_estimate_num_particles,
            )
            self.exclusive_kl_results[bt]["true_elbos"][-1] = elbo

        resource_allocation_metrics = {"bt2num_selected": bt2num_selected}
        return self.exclusive_kl_results, resource_allocation_metrics

    def minimise_exclusive_kl_repeated_successive_halving(
        self,
        initial_num_iterations,
        num_iterations_per_step,
        num_steps,
        num_particles,
        elbo_estimate_num_particles,
        iwae_num_inner,
    ):
        base_seed = torch.randint(high=int(1e9), size=(1,))[0].item()

        guides_items = list(self.guides.items())
        branching_traces = [k for k, v in guides_items]
        bt2num_selected = {bt: 0 for bt in branching_traces}
        bt2num_optim_steps = {bt: 0 for bt in branching_traces}
        bt2optim_state = {bt: None for bt in branching_traces}
        selected_slps = []
        self.exclusive_kl_results = dict()

        # Initialize set of active models
        all_slps = copy.deepcopy(branching_traces)

        # Calculate number of phases
        for i in range(self.utility_class.num_successive_halving_repetitions):
            active_slps = copy.deepcopy(all_slps)
            num_phases = self.utility_class.calculate_num_phases(len(active_slps))
            selected_slps_in_repetition = []
            for j in range(num_phases):
                selected_slps_in_repetition.append(active_slps)
                # Calculate number of time steps in this phase
                num_optimization_steps = (
                    self.utility_class.calculate_num_optimization_steps(
                        len(active_slps), num_phases
                    )
                )

                # Run optimization on each SLP
                args = [
                    (
                        ix,
                        self.prototype_traces[selected_bt],
                        selected_bt,
                        self.branching_sample_values[selected_bt],
                        self.slps_identified_by_discrete_samples,
                        self.guides[selected_bt],
                        self.model,
                        self.learning_rate,
                        bt2optim_state[selected_bt],
                        num_optimization_steps,
                        num_particles,
                        self.smoothing_epsilon,
                        self.estimate_local_elbo,
                        # elbo_estimate_num_particles,
                        num_particles,
                        iwae_num_inner,
                        base_seed,
                        self.save_metrics_every_n,
                    )
                    for ix, selected_bt in enumerate(active_slps)
                ]
                if self.num_parallel_processes > 1:
                    with multiprocessing.get_context("spawn").Pool(
                        self.num_parallel_processes
                    ) as p:
                        results = list(  # Turn iterator into list.
                            tqdm.tqdm(  # Progress bar.
                                p.imap(  # Use imap to return results as soon as they are finished
                                    inner_loop_star,
                                    args,
                                ),
                                total=len(
                                    args
                                ),  # Pass in total length for proper progress bar.
                            )
                        )
                else:
                    results = list(  # Turn iterator into list.
                        tqdm.tqdm(
                            map(
                                inner_loop_star,
                                args,
                            ),
                            total=len(
                                args
                            ),  # Pass in total length for proper progress bar.
                        )
                    )

                # Update metrics
                for ix, selected_bt in enumerate(active_slps):
                    metrics, _, optim_state = results[ix]
                    bt2optim_state[selected_bt] = optim_state
                    bt2num_selected[selected_bt] += 1
                    bt2num_optim_steps[selected_bt] += num_optimization_steps

                    # Append metrics
                    if selected_bt in self.exclusive_kl_results:
                        self.exclusive_kl_results[selected_bt] = {
                            k: v + metrics[k]
                            for k, v in self.exclusive_kl_results[selected_bt].items()
                        }
                    else:
                        self.exclusive_kl_results[selected_bt] = metrics

                # For every SLP that was not selected all the metrics stay fixed. To keep
                # all the metric arrays the same length we repeat the last value for each
                # metric for as many iterations as we run inference on the single SLP.
                num_metrics_saved = len(metrics["losses"])
                for other_bt in self.branching_traces:
                    if other_bt in active_slps:
                        continue

                    self.exclusive_kl_results[other_bt] = {
                        k: v + num_metrics_saved * [v[-1]]
                        for k, v in self.exclusive_kl_results[other_bt].items()
                    }

                # Remove bottom half of of all SLPs
                bt2elbo = {
                    bt: self.exclusive_kl_results[bt]["true_elbos"][-1].item()
                    for bt in active_slps
                }
                active_slps = self.utility_class.select_active_slps(
                    bt2elbo, bt2num_optim_steps
                )

            selected_slps.append(selected_slps_in_repetition)

        # Final estimation of the ELBO
        for bt in all_slps:
            # Get model,guide,branching_trace,branching_sampled_values,
            guide = self.guides[bt]
            elbo, _, _ = estimate_local_elbo_and_iwae(
                self.model,
                guide,
                bt,
                self.branching_sample_values[bt],
                num_monte_carlo=self.elbo_estimate_num_particles,
            )
            self.exclusive_kl_results[bt]["true_elbos"][-1] = elbo

        resource_allocation_metrics = {
            "bt2num_selected": bt2num_selected,
            "bt2num_optim_steps": bt2num_optim_steps,
            "selected_slps": selected_slps,
        }
        return self.exclusive_kl_results, resource_allocation_metrics

    def run(self, forward_kl_callback):
        self.find_slps(self.find_slp_samples)
        logging.info(f"Smoothing epsilon: {self.smoothing_epsilon}")

        self.initialise_guides()

        forward_kl_results = self.minimise_forward_kl(
            self.forward_kl_iter, num_particles=self.forward_kl_num_particles
        )
        if forward_kl_results == dict():
            logging.info(
                "Skipped forward KL phase. All SLPs can be identified by conditioning."
            )
        else:
            forward_kl_callback(self, forward_kl_results, self.model)

        if isinstance(self.utility_class, SuccessiveHalving):
            (
                exclusive_kl_results,
                resource_allocation_metrics,
            ) = self.minimise_exclusive_kl_successive_halving(
                self.initial_num_iterations,
                self.num_iterations_per_step,
                self.num_steps,
                num_particles=self.exclusive_kl_num_particles,
                elbo_estimate_num_particles=self.elbo_estimate_num_particles,
                iwae_num_inner=self.iwae_num_inner,
            )
        elif isinstance(self.utility_class, RepeatedSuccessiveHalving):
            (
                exclusive_kl_results,
                resource_allocation_metrics,
            ) = self.minimise_exclusive_kl_repeated_successive_halving(
                self.initial_num_iterations,
                self.num_iterations_per_step,
                self.num_steps,
                num_particles=self.exclusive_kl_num_particles,
                elbo_estimate_num_particles=self.elbo_estimate_num_particles,
                iwae_num_inner=self.iwae_num_inner,
            )
        else:
            raise ValueError("Unkown utility class")

        return forward_kl_results, exclusive_kl_results, resource_allocation_metrics

    def estimate_local_elbo(self, guide, branching_trace, num_monte_carlo=1):
        log_p = torch.zeros((num_monte_carlo))
        log_q = torch.zeros((num_monte_carlo))
        for i in range(num_monte_carlo):
            with torch.no_grad():
                guide_trace = poutine.trace(guide).get_trace()

                log_q[i] = guide_trace.log_prob_sum()
                log_p[i] = LogJointBranchingTraceHandler(
                    poutine.replay(
                        poutine.condition(
                            self.model,
                            data=self.branching_sample_values[branching_trace],
                        ),
                        trace=guide_trace,
                    ),
                    branching_trace,
                ).log_prob()

        # If the density is -Inf then the given point lies outside the SLP.
        is_in_slp = (~torch.isinf(log_p)).float()
        Z_hat = is_in_slp.mean()

        vals = (is_in_slp / Z_hat) * (log_p - log_q + torch.log(Z_hat))
        # For values generated outside the support we will multiple 0 * -inf which will
        # result in nan's. We want to ignore all those values and therefore only take
        # the mean over the non-nan values. This implicitly assumes that we have no
        # NaN's due to reasons other than values lying outside the SLP support.
        return vals[~vals.isnan()].mean()

    def calculate_slp_weights(self):
        # Input: Dict bt => [elbo]
        # Output: Dict bt => [weight], tensor [log normalization_constant]

        if self.exclusive_kl_results is None:
            raise RuntimeError(
                "Need to run 'minimise_exclusive_kl' before running this function."
            )

        estimator_name = "iwaes" if self.use_iwae_for_weights else "true_elbos"

        branching_traces = sorted(list(self.exclusive_kl_results.keys()))
        num_iterations = len(
            self.exclusive_kl_results[branching_traces[0]][estimator_name]
        )

        # Create matrix of ELBOs
        weights = torch.zeros((len(branching_traces), num_iterations))
        for ix, bt in enumerate(branching_traces):
            weights[ix, :] = torch.tensor(self.exclusive_kl_results[bt][estimator_name])

        normalization_constants = torch.logsumexp(weights, dim=0)
        weights = weights - normalization_constants

        self.bt2weight = dict()
        for ix, bt in enumerate(branching_traces):
            self.bt2weight[bt] = torch.exp(weights[ix, :])

        return self.bt2weight, normalization_constants

    def sample_posterior_predictive(self, num_samples):
        # Create an unconditioned model.
        unconditioned_model = poutine.uncondition(self.model)

        # Create distribution over branching traces.
        bt_and_weights = list(self.bt2weight.items())
        branching_traces = [bt for bt, _ in bt_and_weights]
        slp_dist = dist.Categorical(
            torch.tensor([weight[-1] for _, weight in bt_and_weights])
        )

        posterior_samples = []

        for _ in range(num_samples):
            # Sample a branching trace
            slp_ix = slp_dist.sample()
            bt = branching_traces[slp_ix]

            # Condition on all the sampled values
            slp_model = poutine.condition(
                unconditioned_model, data=self.branching_sample_values[bt]
            )

            is_outside_slp = True
            num_tries = 0
            with torch.no_grad():
                while is_outside_slp:
                    num_tries += 1
                    # Sample from the guide
                    guide_trace = poutine.trace(self.guides[bt]).get_trace()

                    is_outside_slp = torch.isinf(
                        LogJointBranchingTraceHandler(
                            poutine.replay(slp_model, trace=guide_trace), bt
                        ).log_prob()
                    )

                # Replay the model
                posterior_sample = poutine.trace(
                    poutine.replay(slp_model, trace=guide_trace)
                ).get_trace()

            # Repeat the sample `num_tries` times so that when we take means we
            # get the correct normalization constant i.e. we normalize by the
            # number of times that we have actually sampled.
            samples = num_tries * [posterior_sample]
            posterior_samples += samples

        return posterior_samples


def estimate_iwae(guide, model, num_particles):
    iwae_terms = torch.zeros(num_particles)
    for ix in range(num_particles):
        guide_trace = poutine.trace(guide).get_trace()

        log_q = guide_trace.log_prob_sum()

        # Evaluate smoothed model log probability
        log_p = (
            poutine.trace(
                poutine.replay(
                    model,
                    trace=guide_trace,
                )
            )
            .get_trace()
            .log_prob_sum()
        )

        iwae_terms[ix] = log_p.detach() - log_q.detach()

    return torch.logsumexp(iwae_terms, dim=0) - torch.log(torch.tensor(num_particles))


def inner_loop_star(args):
    return inner_loop(*args)


def inner_loop(*args, **kwargs):
    (
        worker_ix,
        prototype_trace,
        branching_trace,
        branching_sampled_values,
        slps_identified_by_discrete_samples,
        guide,
        model,
        learning_rate,
        optim_state,
        num_iterations,
        num_particles,
        smoothing_epsilon,
        estimate_local_elbo,
        elbo_estimate_num_particles,
        iwae_num_inner,
        base_seed,
        save_metrics_every_n,
    ) = args
    # if slp_identified_by_sampled_values(branching_trace, branching_sampled_values):
    if slps_identified_by_discrete_samples:
        return inner_loop_reparam(*args, **kwargs)
    else:
        return inner_loop_score(*args, **kwargs)


def inner_loop_reparam(
    worker_ix,
    prototype_trace,
    branching_trace,
    branching_sampled_values,
    slps_identified_by_discrete_samples,
    guide,
    model,
    learning_rate,
    optim_state,
    num_iterations,
    num_particles,
    smoothing_epsilon,
    estimate_local_elbo,
    elbo_estimate_num_particles,
    iwae_num_inner,
    base_seed,
    save_every,
):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(base_seed + worker_ix)
    pyro.set_rng_seed(base_seed + worker_ix)

    # save_every = 200

    optimizer = torch.optim.Adam(guide.parameters(), lr=learning_rate)
    if not (optim_state is None):
        optimizer.load_state_dict(optim_state)

    losses = []
    smoothed_elbos = []
    true_elbos = []
    iwaes = []
    weight_variances = []
    gradient_norms = dict()
    gradient_norm_global = dict()
    params = dict()
    for name, p in guide.named_parameters():
        gradient_norms[f"{name}_grad_norm"] = []
        params[name] = [p.tolist()]
        gradient_norm_global[name] = None

        p.register_hook(
            lambda g, name=name: gradient_norm_global.update({name: g.norm()})
        )

    conditioned_model = poutine.condition(model, data=branching_sampled_values)

    for i in range(num_iterations):
        optimizer.zero_grad()

        loss = torch.tensor(0.0)
        smoothed_elbo = 0.0
        iwae_terms = torch.zeros(num_particles)
        for ix in range(num_particles):
            guide_trace = poutine.trace(guide).get_trace()

            log_q = guide_trace.log_prob_sum()

            # Evaluate smoothed model log probability
            log_p = (
                poutine.trace(
                    poutine.replay(
                        conditioned_model,
                        trace=guide_trace,
                    )
                )
                .get_trace()
                .log_prob_sum()
            )

            # We do not want to backprop through the first log_q and therefore .detach().
            loss += -(log_p - log_q) / num_particles
            smoothed_elbo += (log_p.detach() - log_q.detach()) / num_particles
            iwae_terms[ix] = log_p.detach() - log_q.detach()

        loss.backward()

        optimizer.step()

        # Special case for first iteration.
        if ((i + 1) % save_every) == 0 or i == 0:
            losses.append(loss.detach())
            smoothed_elbos.append(smoothed_elbo)
            # For reparameterized gradient the smoothed ELBO is equivalent to the true
            # ELBO and should have reasonably low variance.
            true_elbos.append(smoothed_elbo)
            iwaes.append(
                torch.logsumexp(iwae_terms, dim=0)
                - torch.log(torch.tensor(num_particles))
            )
            weight_variances.append(torch.var(iwae_terms))
            for n, p in guide.named_parameters():
                params[n].append(p.tolist())
                gradient_norms[f"{n}_grad_norm"].append(gradient_norm_global[n])

    # Final evaluation of metrics.
    loss = torch.tensor(0.0)
    smoothed_elbo = 0.0
    iwae_terms = torch.zeros(elbo_estimate_num_particles)
    if hasattr(model, "num_subsample"):
        # This is a hack at the moment to not do any subsampling when doing the final ELBO evaluation.
        # Something more principled should probably be using effect handlers.
        new_model = copy.deepcopy(model)
        new_model.num_subsample = None
        conditioned_model = poutine.condition(new_model, data=branching_sampled_values)

    for ix in range(elbo_estimate_num_particles):
        guide_trace = poutine.trace(guide).get_trace()

        log_q = guide_trace.log_prob_sum()

        # Evaluate smoothed model log probability
        log_p = (
            poutine.trace(
                poutine.replay(
                    conditioned_model,
                    trace=guide_trace,
                )
            )
            .get_trace()
            .log_prob_sum()
        )

        # We do not want to backprop through the first log_q and therefore .detach().
        loss += -(log_p - log_q) / elbo_estimate_num_particles
        smoothed_elbo += (log_p.detach() - log_q.detach()) / elbo_estimate_num_particles
        iwae_terms[ix] = log_p.detach() - log_q.detach()

    losses.append(loss.detach())
    smoothed_elbos.append(smoothed_elbo)
    # For reparameterized gradient the smoothed ELBO is equivalent to the true
    # ELBO and should have reasonably low variance.
    true_elbos.append(smoothed_elbo)
    iwaes.append(
        torch.logsumexp(iwae_terms, dim=0)
        - torch.log(torch.tensor(elbo_estimate_num_particles))
    )
    weight_variances.append(torch.var(iwae_terms))

    metrics = {
        "losses": losses,
        "smoothed_elbos": smoothed_elbos,
        "true_elbos": true_elbos,
        "iwaes": iwaes,
        "weight_variances": weight_variances,
    }
    return metrics | params | gradient_norms, iwae_terms, optimizer.state_dict()


def estimate_local_elbo_and_iwae(
    model,
    guide,
    branching_trace,
    branching_sampled_values,
    num_monte_carlo=1,
    iwae_num_inner=None,
):
    if iwae_num_inner is None:
        iwae_num_inner = num_monte_carlo

    log_p = torch.zeros((num_monte_carlo))
    log_q = torch.zeros((num_monte_carlo))
    for i in range(num_monte_carlo):
        with torch.no_grad():
            guide_trace = poutine.trace(guide).get_trace()

            log_q[i] = guide_trace.log_prob_sum()
            log_p[i] = LogJointBranchingTraceHandler(
                poutine.replay(
                    poutine.condition(
                        model,
                        data=branching_sampled_values,
                    ),
                    trace=guide_trace,
                ),
                branching_trace,
            ).log_prob()

    # If the density is -Inf then the given point lies outside the SLP.
    is_in_slp = (~torch.isinf(log_p)).float()
    Z_hat = is_in_slp.mean()

    elbos = (is_in_slp / Z_hat) * (log_p - log_q + torch.log(Z_hat))
    # For values generated outside the support we will multiple 0 * -inf which will
    # result in nan's. We want to ignore all those values and therefore only take
    # the mean over the non-nan values. This implicitly assumes that we have no
    # NaN's due to reasons other than values lying outside the SLP support.
    if all(elbos.isnan()):
        elbo = torch.tensor(float("-inf"))
    else:
        elbo = elbos[~elbos.isnan()].sum() / num_monte_carlo

    log_weights = log_p - log_q
    iwae_num_outer = int(num_monte_carlo / iwae_num_inner)
    log_weights = log_weights.reshape((iwae_num_outer, iwae_num_inner))
    iwae_inner = torch.logsumexp(log_weights, dim=1) - torch.log(
        torch.tensor(iwae_num_inner)
    )
    iwae = torch.mean(iwae_inner)
    return elbo, iwae, log_weights


def inner_loop_score(
    worker_ix,
    prototype_trace,
    branching_trace,
    branching_sampled_values,
    slps_identified_by_discrete_samples,
    guide,
    model,
    learning_rate,
    optim_state,
    num_iterations,
    num_particles,
    smoothing_epsilon,
    estimate_local_elbo,
    elbo_estimate_num_particles,
    iwae_num_inner,
    base_seed,
    save_every,
):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(base_seed + worker_ix)
    pyro.set_rng_seed(base_seed + worker_ix)

    optimizer = torch.optim.Adam(guide.parameters(), lr=learning_rate)
    if not (optim_state is None):
        optimizer.load_state_dict(optim_state)

    params = {name: [p.tolist()] for name, p in guide.named_parameters()}
    losses = []
    smoothed_elbos = []
    true_elbos = []
    iwaes = []

    for i in range(num_iterations):
        optimizer.zero_grad()

        loss = torch.tensor(0.0)
        smoothed_elbo = 0.0
        for _ in range(num_particles):
            guide_trace = poutine.trace(guide).get_trace()

            # log_q = guide_trace.log_prob_sum()
            log_q = guide.log_joint_trace(guide_trace)

            # Evaluate smoothed model log probability
            log_p = LogJointBranchingTraceHandler(
                poutine.replay(
                    poutine.condition(model, data=branching_sampled_values),
                    trace=guide_trace,
                ),
                branching_trace,
                epsilon=smoothing_epsilon,
            ).log_prob()

            # We do not want to backprop through the first log_q and therefore .detach().
            loss += -((log_p.detach() - log_q.detach()) * log_q) / num_particles
            smoothed_elbo += (log_p.detach() - log_q.detach()) / num_particles

        loss.backward()

        optimizer.step()

        # Special case for first iteration.
        if ((i + 1) % save_every) == 0 or i == 0:
            losses.append(loss.detach())
            smoothed_elbos.append(smoothed_elbo)

            elbo, iwae, log_weights = estimate_local_elbo_and_iwae(
                model,
                guide,
                branching_trace,
                branching_sampled_values,
                num_monte_carlo=elbo_estimate_num_particles,
                iwae_num_inner=iwae_num_inner,
            )
            true_elbos.append(elbo)
            iwaes.append(iwae)
            for n, p in guide.named_parameters():
                params[n].append(p.tolist())

    metrics = {
        "losses": losses,
        "smoothed_elbos": smoothed_elbos,
        "true_elbos": true_elbos,
        "iwaes": iwaes,
    }
    return metrics | params, log_weights, optimizer.state_dict()
