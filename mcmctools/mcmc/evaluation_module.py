import numpy as np


from mcmctools.utils.json import load_json_sim_params


""" To be used to evaluate a given simulation preformed in C++. In particular, in this case, no model related parameters are necessary. """


class EvaluationModule:
    def __init__(self, sim_base_dir="./",
                 rel_data_dir=None,  # -> sim_base_dir + "/" + rel_data_dir
                 rel_results_dir=None,  # -> sim_base_dir + "/" + rel_results_dir
                 running_parameter_kind=None,
                 running_parameter=None,
                 rp_values=None,
                 **kwargs):
        if sim_base_dir is None:
            import os
            self.sim_base_dir = os.getcwd()
        else:
            self.sim_base_dir = sim_base_dir
        self.rel_data_dir = rel_data_dir
        self.rel_results_dir = rel_results_dir
        self.running_parameter_kind = running_parameter_kind
        self.running_parameter = running_parameter
        self.rp_values = rp_values

        self.computed_equilibrium_times = None
        self.computed_correlation_times = None
        self.computed_expectation_values = None

    def compute_equilibrium_time(self, confidence_range=None, confidence_window=None,
                                 fma=None, custom_load_data_func=None, custom_load_data_args=None):
        if self.rel_data_dir is None:
            assert False, "rel_data_dir is undefined"

        sim_params = load_json_sim_params(
            rel_data_dir=self.rel_data_dir, identifier="equilibrium_time",
            running_parameter=self.running_parameter, rp_values=self.rp_values, sim_base_dir=self.sim_base_dir)
        execution_params = sim_params["execution_mode"]
        sample_size = execution_params["sample_size"]
        number_of_steps = execution_params["number_of_steps"]
        measure = execution_params["measure"]

        if confidence_range is None:
            confidence_range = execution_params["confidence_range"]
        if confidence_window is None:
            confidence_window = np.int(execution_params["confidence_window"])

        from mcmctools.modes.equilibrium_time import equilibrium_time
        equilibrium_times = equilibrium_time(
            sample_size=sample_size, number_of_steps=number_of_steps, confidence_range=confidence_range,
            confidence_window=confidence_window, measure=measure,
            running_parameter=self.running_parameter, rp_values=self.rp_values,
            rel_data_dir=self.rel_data_dir, data=None, rel_results_dir=self.rel_results_dir,
            sim_base_dir=self.sim_base_dir, fma=fma, custom_load_data_func=custom_load_data_func,
            custom_load_data_args=custom_load_data_args)
        self.computed_equilibrium_times = equilibrium_times

    def compute_correlation_time(self, fma=None, custom_load_data_func=None, custom_load_data_args=None):
        if self.rel_data_dir is None:
            assert False, "rel_data_dir is undefined"

        sim_params = load_json_sim_params(
            rel_data_dir=self.rel_data_dir, identifier="correlation_time",
            running_parameter=self.running_parameter, rp_values=self.rp_values, sim_base_dir=self.sim_base_dir)
        execution_params = sim_params["execution_mode"]
        minimum_sample_size = execution_params["minimum_sample_size"]
        maximum_correlation_time = execution_params["maximum_correlation_time"]
        measure = execution_params["measure"]

        from mcmctools.modes.correlation_time import correlation_time
        correlation_times = correlation_time(
            minimum_sample_size=minimum_sample_size, maximum_correlation_time=maximum_correlation_time, measure=measure,
            running_parameter=self.running_parameter, rp_values=self.rp_values,
            rel_data_dir=self.rel_data_dir, data=None, rel_results_dir=self.rel_results_dir,
            sim_base_dir=self.sim_base_dir, fma=fma, custom_load_data_func=custom_load_data_func,
            custom_load_data_args=custom_load_data_args)
        self.computed_correlation_times = correlation_times

    def compute_expectation_values(self, measures=None, error_type=None, n_means_bootstrap=None, custom_measures_func=None,
                                  custom_measures_args=None, custom_load_data_func=None, custom_load_data_args=None):
        if self.rel_data_dir is None:
            assert False, "rel_data_dir is undefined"
        sim_params = load_json_sim_params(
            rel_data_dir=self.rel_data_dir, identifier="expectation_value",
            running_parameter=self.running_parameter, rp_values=self.rp_values, sim_base_dir=self.sim_base_dir)
        execution_params = sim_params["execution_mode"]
        number_of_measurements = execution_params["number_of_measurements"]

        if error_type is None:
            error_type = execution_params["error_type"]
        if n_means_bootstrap is None:
            if "n_means_bootstrap" not in execution_params:
                n_means_bootstrap = 0
            else:
                n_means_bootstrap = execution_params["n_means_bootstrap"]

        # Allows the computation of additional measures
        if measures is None:
            measures = execution_params["measures"]

        from mcmctools.modes.expectation_value import expectation_value
        expectation_values = expectation_value(
            number_of_measurements=number_of_measurements, measures=measures,
            error_type=error_type, n_means_bootstrap=n_means_bootstrap, running_parameter=self.running_parameter, rp_values=self.rp_values,
            rel_data_dir=self.rel_data_dir, data=None, rel_results_dir=self.rel_results_dir,
            sim_base_dir=self.sim_base_dir,
            custom_measures_func=custom_measures_func, custom_measures_args=custom_measures_args,
            custom_load_data_func=custom_load_data_func, custom_load_data_args=custom_load_data_args
        )
        self.computed_expectation_values = expectation_values

    def load_equilibrium_times(self):
        if self.rel_results_dir is not None:
            from mcmctools.modes.equilibrium_time import load_equilibrium_times_results
            self.computed_equilibrium_times = load_equilibrium_times_results(rel_results_dir=self.rel_results_dir,
                                                                             sim_base_dir=self.sim_base_dir)
            return self.equilibrium_times
        else:
            assert False, "rel_results_dir is undefined"

    def load_correlation_times(self):
        if self.rel_results_dir is not None:
            from mcmctools.modes.correlation_time import load_correlation_times_results
            self.computed_correlation_times = load_correlation_times_results(rel_results_dir=self.rel_results_dir,
                                                                             sim_base_dir=self.sim_base_dir)
            return self.correlation_times
        else:
            assert False, "rel_results_dir is undefined"

    def load_expectation_values(self):
        if self.rel_results_dir is not None:
            from mcmctools.modes.expectation_value import load_expectation_value_results
            self.computed_expectation_values = load_expectation_value_results(rel_results_dir=self.rel_results_dir,
                                                                              sim_base_dir=self.sim_base_dir)
            return self.expectation_values
        else:
            assert False, "rel_results_dir is undefined"

    def load_data(self, identifier="expectation_value", custom_load_data_func=None, custom_load_data_args=None):
        from mcmctools.loading.loading import load_data_based_running_parameter
        data = load_data_based_running_parameter(
            rel_data_dir=self.rel_data_dir, identifier=identifier, running_parameter=self.running_parameter,
            rp_values=self.rp_values,
            sim_base_dir=self.sim_base_dir, custom_load_data_func=custom_load_data_func,
            custom_load_data_args=custom_load_data_args)
        return data

    def load_sim_params(self, identifier = "expectation_value"):
        sim_params = load_json_sim_params(
        rel_data_dir = self.rel_data_dir, identifier = identifier,
        running_parameter = self.running_parameter, rp_values = self.rp_values, sim_base_dir = self.sim_base_dir)
        return sim_params

    @property
    def equilibrium_times(self):
        return self.computed_equilibrium_times

    @property
    def correlation_times(self):
        return self.computed_correlation_times

    @property
    def expectation_values(self):
        return self.computed_expectation_values