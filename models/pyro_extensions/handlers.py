import torch
import pyro
import copy

from collections import OrderedDict

from .util import get_sample_addresses


class BranchingTraceMessenger(pyro.poutine.messenger.Messenger):

    def __init__(self):
        super().__init__()

    def __call__(self, fn):
        return BranchingTraceHandler(self, fn)

    def __enter__(self):
        self.branching_trace = ""
        self.sampled_values = OrderedDict()
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        return super().__exit__(*args, **kwargs)

    def _postprocess_message(self, msg):
        if msg["type"] == "sample" and msg["infer"].get("branching", False):
            self.branching_trace += str(msg["value"].item())
            self.sampled_values[msg["name"]] = msg["value"]

    def get_trace(self):
        return copy.deepcopy(self.branching_trace)

    def get_sampled_values(self):
        return copy.deepcopy(self.sampled_values)


class BranchingTraceHandler:
    def __init__(self, msngr, fn):
        self.fn = fn
        self.msngr = msngr

    def __call__(self, *args, **kwargs):
        with self.msngr:
            return self.fn(*args, **kwargs)

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.msngr.get_trace()


_, branching_trace = pyro.poutine.handlers._make_handler(BranchingTraceMessenger)


class LogJointBranchingTraceHandler:

    def __init__(self, fn, branching_trace, epsilon=torch.tensor(float("-inf"))):
        self.fn = fn
        self.msngr = BranchingTraceMessenger()
        self.trace_msngr = pyro.poutine.trace_messenger.TraceMessenger()
        self.slp_identifier = branching_trace
        self.epsilon = epsilon

    def __call__(self, *args, **kwargs):
        with self.trace_msngr:
            with self.msngr:
                return self.fn(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        self(*args, **kwargs)
        trace = self.trace_msngr.get_trace()
        if "," in self.slp_identifier:
            slp_id = ",".join(get_sample_addresses(trace))
        else:
            slp_id = self.msngr.get_trace()

        if slp_id == self.slp_identifier:
            return trace.log_prob_sum()
        else:
            return self.epsilon