import pyro
import torch
import sys
import os
import numpy as np

HOME_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(HOME_DIR)
from models.gmm import FiniteGMMModel

OUT_FOLDER = "data/gmm/"

SEED = 0

NUM_CLUSTERS = 5

NUM_HELD_OUT_DATA_POINTS = 500

DIMENSIONALITY = 100
NUM_OBSERVATIONS = 1000

def main():
    pyro.set_rng_seed(SEED)

    base_model = FiniteGMMModel(
        cluster_means_dim=DIMENSIONALITY, num_observations=NUM_OBSERVATIONS
    )

    # We set NUM_CLUSTERS - 1 because the Categorical distribution has support
    # {0, ..., K} so it starts at 0.
    model = pyro.poutine.block(
        pyro.condition(
            base_model,
            {"num_clusters": torch.tensor(NUM_CLUSTERS - 1)},
        ),
        hide=["num_clusters"],
    )
    trace = pyro.poutine.trace(model).get_trace()
    data = {"num_clusters": NUM_CLUSTERS}
    param_names = ["obs"]
    for i in range(NUM_CLUSTERS):
        param_names.append(f"mean_{i}")

    for name in param_names:
        data[name] = trace.nodes[name]["value"].numpy() / trace.nodes["obs"]["value"].numpy().std()

    fname_prefix = f"finite_gmm_data_n={base_model.num_observations}_mean_dim={base_model.cluster_means_dim}"
    np.savez(os.path.join(OUT_FOLDER, f"{fname_prefix}.npz"), **data)

    # Generate held-out data for evaluation.
    cluster_means = {
        f"mean_{i}": trace.nodes[f"mean_{i}"]["value"] for i in range(NUM_CLUSTERS)
    }
    val_trace = pyro.poutine.trace(
        pyro.condition(model, data=cluster_means)
    ).get_trace()
    data = {"num_clusters": NUM_CLUSTERS}
    for i in range(NUM_CLUSTERS):
        assert all(
            val_trace.nodes[f"mean_{i}"]["value"] == trace.nodes[f"mean_{i}"]["value"]
        )

    data["obs"] = (
        val_trace.nodes["obs"]["value"].numpy()[:NUM_HELD_OUT_DATA_POINTS]
        / trace.nodes["obs"]["value"].numpy().std()
    )

    fname_prefix = f"finite_gmm_data_n={NUM_HELD_OUT_DATA_POINTS}_mean_dim={base_model.cluster_means_dim}"
    np.savez(os.path.join(OUT_FOLDER, f"{fname_prefix}_validation.npz"), **data)


if __name__ == "__main__":
    main()