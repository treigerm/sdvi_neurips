from pyro import poutine


def get_sample_addresses(trace: poutine.Trace, included_addresses: set = set()):
    addresses = []
    for name, site in trace.nodes.items():
        if (
            site["type"] != "sample"
            or site["is_observed"]
            # When doing posterior predictive simulation we want to ignore what used to be observe
            or site.get("infer", dict()).get("was_observed", False)
            or poutine.util.site_is_subsample(site)
        ) and not (
            (name in included_addresses)
            #  Always want to include branching addresses.
            or site.get("infer", dict()).get("branching", False)
        ):
            continue
        addresses.append(name)

    return addresses