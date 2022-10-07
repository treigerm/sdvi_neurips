import math
import heapq
import torch
import pyro.distributions as dist


class AbstractUtility:
    def utility_function(self, bt2log_weights, bt2num_selected, max_weight, **kwargs):
        raise NotImplementedError()


class DCCUtility(AbstractUtility):
    """See https://arxiv.org/abs/1910.13324 Appendix F."""

    def __init__(self, delta=0.5, beta=1, kappa=0.1, num_lookahead=1000):
        self.delta = delta
        self.beta = beta
        self.kappa = kappa
        self.num_lookahead = num_lookahead

    @staticmethod
    def exploitation_term(log_weights, kappa):
        iwae_est = torch.logsumexp(log_weights, dim=0) - torch.log(
            torch.tensor(log_weights.shape[0])
        )
        Z_est = torch.exp(iwae_est)

        weight_variance = torch.var(torch.exp(log_weights), unbiased=True)
        return torch.sqrt(torch.pow(Z_est, 2) + (1 + kappa) * weight_variance)

    @staticmethod
    def exploration_weight_term(log_weights, threshold_weight, num_lookahead):
        num_inf = torch.isinf(log_weights).sum()
        if log_weights.size(0) - num_inf <= 2:
            return 0.0

        log_weights = log_weights[~torch.isinf(log_weights)]

        weight_mean = torch.mean(log_weights)
        weight_std = torch.std(log_weights, unbiased=True)

        # Convert to 64-bit Floats and work in log domain for higher numerical precision.
        weight_dist = dist.Normal(weight_mean.double(), weight_std.double())
        log_threshold_exceed_prob = num_lookahead * torch.log(
            weight_dist.cdf(threshold_weight)
        )
        return 1.0 - torch.exp(log_threshold_exceed_prob)

    def utility_function(self, bt2log_weights, bt2num_selected, max_weight):
        utilities = dict()

        branching_traces = list(bt2log_weights.keys())
        num_slps = len(branching_traces)
        exploitation_terms = torch.zeros((num_slps,))
        exploration_weight_terms = torch.zeros((num_slps,))
        pure_exploration_terms = torch.zeros((num_slps,))

        complete_selected = torch.log(
            torch.tensor(sum([v for v in bt2num_selected.values()]))
        )
        for ix in range(num_slps):
            bt = branching_traces[ix]

            pure_exploration_terms[ix] = complete_selected

            log_weights = bt2log_weights[bt]
            exploitation_terms[ix] = self.exploitation_term(log_weights, self.kappa)
            exploration_weight_terms[ix] = self.exploration_weight_term(
                log_weights, max_weight, self.num_lookahead
            )
            pure_exploration_terms[ix] /= torch.sqrt(torch.tensor(bt2num_selected[bt]))

        # Normalize all the terms
        max_exploitation_terms = torch.max(exploitation_terms)
        if max_exploitation_terms == 0.0 or any(torch.isnan(exploitation_terms)):
            # Avoid division by 0 or nans.
            exploitation_terms[:] = 1.0
        else:
            exploitation_terms /= max_exploitation_terms

        max_exploration_weight_terms = torch.max(exploration_weight_terms)
        if max_exploration_weight_terms == 0.0:
            exploration_weight_terms[:] = 1.0
        else:
            exploration_weight_terms /= max_exploration_weight_terms

        if any(torch.isnan(exploration_weight_terms)):
            original_terms = [
                self.exploration_weight_term(
                    bt2log_weights[branching_traces[ix]], max_weight, self.num_lookahead
                )
                for ix in range(num_slps)
            ]
            raise RuntimeError(
                f"Detected nan's in exploration weight term ({exploration_weight_terms}), original weights ({original_terms})"
            )
        for ix, bt in enumerate(branching_traces):
            S_k = bt2num_selected[bt]
            utilities[bt] = (1.0 / S_k) * (
                (1.0 - self.delta) * exploitation_terms[ix]
                + self.delta * exploration_weight_terms[ix]
                + self.beta * pure_exploration_terms[ix]
            )

        print(branching_traces)
        print(exploitation_terms)
        print(exploration_weight_terms)
        print(pure_exploration_terms)
        return utilities


class SuccessiveHalving:
    def __init__(
        self, num_total_iterations: int, num_final_arms: int = 10
    ):
        self.num_total_iterations = num_total_iterations
        self.num_final_arms = num_final_arms

    def calculate_num_phases(self, num_arms: int) -> int:
        self.num_phases = 1
        num_active_arms = num_arms
        while num_active_arms > self.num_final_arms:
            num_active_arms = max(math.ceil(num_active_arms / 2), self.num_final_arms)
            self.num_phases += 1

        return self.num_phases

    def calculate_num_optimization_steps(self, num_active_arms: int) -> int:
        return math.floor(
            self.num_total_iterations / (self.num_phases * num_active_arms)
        )

    def select_active_slps(self, arm2reward: dict[str, float]) -> list[str]:
        num_active_arms = len(arm2reward.keys())
        num_to_keep = max(math.ceil(num_active_arms / 2), self.num_final_arms)
        arms_to_keep = heapq.nlargest(num_to_keep, arm2reward, key=arm2reward.get)
        return arms_to_keep


class RepeatedSuccessiveHalving:
    def __init__(
        self,
        num_total_iterations: int,
        num_sh_repetitions: int,
        num_final_arms: int,
        alpha: float,
    ):
        self.total_optimization_steps = num_total_iterations
        self.num_successive_halving_repetitions = num_sh_repetitions
        self.num_final_arms = num_final_arms
        self.alpha = alpha

        # Total optimization budget for each repetition of successive halving.
        self.successive_halving_budget = math.floor(
            num_total_iterations / num_sh_repetitions
        )

    def calculate_num_phases(self, num_arms: int) -> int:
        self.num_phases = 1
        num_active_arms = num_arms
        while num_active_arms > self.num_final_arms:
            num_active_arms = max(math.ceil(num_active_arms / 2), self.num_final_arms)
            self.num_phases += 1

        return self.num_phases

    def calculate_num_optimization_steps(
        self, num_active_arms: int, num_phases: int
    ) -> int:
        return math.floor(
            self.successive_halving_budget / (num_phases * num_active_arms)
        )

    def select_active_slps(
        self, arm2elbo: dict[str, float], arm2num_optim_steps: dict[str, int]
    ) -> list[str]:
        arm2reward = dict()
        for arm in arm2elbo.keys():
            arm2reward[arm] = self.alpha * arm2elbo[arm] - math.log(
                arm2num_optim_steps[arm]
            )

        num_active_arms = len(arm2reward.keys())
        num_to_keep = max(math.ceil(num_active_arms / 2), self.num_final_arms)
        arms_to_keep = heapq.nlargest(num_to_keep, arm2reward, key=arm2reward.get)

        return arms_to_keep