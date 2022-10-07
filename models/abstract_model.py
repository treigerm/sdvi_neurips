class AbstractModel:

    autoguide_hide_vars = []

    does_lppd_evaluation = False

    slps_identified_by_discrete_samples = False

    def make_parameter_plots(self, results, guide, branching_trace, file_prefix):
        raise NotImplementedError()

    def calculate_ground_truth_weights(self, sdvi):
        return None, None