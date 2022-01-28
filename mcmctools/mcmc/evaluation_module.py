import numpy as np


""" To be used to evaluate a given simulation preformed in C++. In particular, in this case, no model related parameters are necessary. """


class EvaluationModule:
    def __init__(self, sim_base_dir=None,
                 rel_data_path=None,  # -> sim_base_dir + "/" + rel_data_path
                 rel_results_path=None,  # -> sim_base_dir + "/" + rel_results_path
                 running_parameter_kind=None,
                 running_parameter=None,
                 rp_values=None,
                 **kwargs):
        if sim_base_dir is None:
            import os
            self.sim_base_dir = os.getcwd()
        else:
            self.sim_base_dir = sim_base_dir
        self.rel_data_path = rel_data_path
        self.rel_results_path = rel_results_path
        self.running_parameter_kind = running_parameter_kind
        self.running_parameter = running_parameter
        self.rp_values = rp_values

        self.computed_equilibrium_times = None
        self.computed_correlation_times = None
        self.computed_expectation_values = None

    def compute_equilibrium_time(self, data=None, sample_size=None, number_of_steps=None, eval_confidence_range=0.1,
                                 eval_confidence_window=10, measure=None, rp_values=None, from_file=False, fma=None,
                                 custom_load_data_func=None, custom_load_data_args=None):
        if rp_values is None:
            rp_values = self.rp_values

        if from_file:
            assert self.rel_data_path is not None, "simulation config file cannot be found if rel_data_path is not defined."
            from mcmctools.utils.json import load_json_sim_params
            sim_params = load_json_sim_params(
                rel_data_path=self.rel_data_path, identifier="equilibrium_time",
                running_parameter=self.running_parameter, rp_key=rp_values[0], sim_base_dir=self.sim_base_dir)
            execution_params = sim_params["execution_params"]
            sample_size = execution_params["sample_size"]
            number_of_steps = execution_params["number_of_steps"]
            eval_confidence_range = execution_params["confidence_range"]
            eval_confidence_window = np.int(execution_params["confidence_window"])
        else:
            assert data is not None, "data needs to be defined if not loaded from file"
            assert sample_size is not None, "data needs to be defined if not loaded from file"
            assert number_of_steps is not None, "data needs to be defined if not loaded from file"

        from mcmctools.modes.equilibrium_time import equilibrium_time
        equilibrium_times = equilibrium_time(
            sample_size=sample_size, number_of_steps=number_of_steps, confidence_range=eval_confidence_range,
            confidence_window=eval_confidence_window, measure=measure,
            running_parameter=self.running_parameter, rp_values=rp_values,
            rel_data_dir=self.rel_data_path, data=data, rel_results_dir=self.rel_results_path,
            sim_base_dir=self.sim_base_dir, fma=fma, custom_load_data_func=custom_load_data_func,
            custom_load_data_args=custom_load_data_args)
        self.computed_equilibrium_times = equilibrium_times

    def compute_correlation_time(self, data=None, minimum_sample_size=None, maximum_correlation_time=None, measure=None,
                                 rp_values=None, from_file=False, fma=None, custom_load_data_func=None, custom_load_data_args=None):
        if rp_values is None:
            rp_values = self.rp_values

        if from_file:
            assert self.rel_data_path is not None, "simulation config file cannot be found if rel_data_path is not defined."
            from mcmctools.utils.json import load_json_sim_params
            sim_params = load_json_sim_params(
                rel_data_path=self.rel_data_path, identifier="correlation_time",
                running_parameter=self.running_parameter, rp_key=rp_values[0], sim_base_dir=self.sim_base_dir)
            execution_params = sim_params["execution_params"]
            minimum_sample_size = execution_params["minimum_sample_size"]
            maximum_correlation_time = execution_params["maximum_correlation_time"]
            measure = execution_params["measure"]

        from mcmctools.modes.correlation_time import correlation_time
        correlation_times = correlation_time(
            minimum_sample_size=minimum_sample_size, maximum_correlation_time=maximum_correlation_time, measure=measure,
            running_parameter=self.running_parameter, rp_values=rp_values,
            rel_data_dir=self.rel_data_path, data=data, rel_results_dir=self.rel_results_path,
            sim_base_dir=self.sim_base_dir, fma=fma, custom_load_data_func=custom_load_data_func,
            custom_load_data_args=custom_load_data_args)
        self.computed_correlation_times = correlation_times

    # Eval measures can be different to the simulation, the others not (number_of_measurements as well..??)
    def compute_expectation_value(self, data=None, number_of_measurements=None, measures=None,
                                  eval_error_type="statistical",  eval_n_means_bootstrap=0, rp_values=None, from_file=False, custom_measures_func=None,
                                  custom_measures_args=None, custom_load_data_func=None, custom_load_data_args=None):
        if rp_values is None:
            rp_values = self.rp_values

        if from_file:
            assert self.rel_data_path is not None, "simulation config file cannot be found if rel_data_path is not defined."
            from mcmctools.utils.json import load_json_sim_params
            sim_params = load_json_sim_params(
                rel_data_path=self.rel_data_path, identifier="expectation_value",
                running_parameter=self.running_parameter, rp_key=rp_values[0], sim_base_dir=self.sim_base_dir)
            execution_params = sim_params["execution_params"]
            measures = execution_params["measures"]
            eval_error_type = execution_params["error_type"]
            eval_n_means_bootstrap = execution_params["n_means_bootstrap"]
            number_of_measurements = execution_params["number_of_measurements"]

        from mcmctools.modes.expectation_value import expectation_value
        expectation_values = expectation_value(
            number_of_measurements=number_of_measurements, measures=measures,
            error_type=eval_error_type, n_means_bootstrap=eval_n_means_bootstrap, running_parameter=self.running_parameter, rp_values=rp_values,
            rel_data_dir=self.rel_data_path, data=data, rel_results_dir=self.rel_results_path,
            sim_base_dir=self.sim_base_dir,
            custom_measures_func=custom_measures_func, custom_measures_args=custom_measures_args,
            custom_load_data_func=custom_load_data_func, custom_load_data_args=custom_load_data_args
        )
        self.computed_expectation_values = expectation_values

    def load_equilibrium_time_from_file(self):
        if self.rel_results_path is not None:
            from mcmctools.modes.equilibrium_time import load_equilibrium_times_results
            self.computed_equilibrium_times = load_equilibrium_times_results(rel_results_dir=self.rel_results_path,
                                                                             sim_base_dir=self.sim_base_dir)
            return self.equilibrium_times
        else:
            assert False, "rel_results_path is undefined"

    def load_correlation_time_from_file(self):
        if self.rel_results_path is not None:
            from mcmctools.modes.correlation_time import load_correlation_times_results
            self.computed_correlation_times = load_correlation_times_results(rel_results_dir=self.rel_results_path,
                                                                             sim_base_dir=self.sim_base_dir)
            return self.correlation_times
        else:
            assert False, "rel_results_path is undefined"

    def load_expectation_values_from_file(self):
        if self.rel_results_path is not None:
            from mcmctools.modes.expectation_value import load_expectation_value_results
            self.computed_expectation_values = load_expectation_value_results(rel_results_dir=self.rel_results_path,
                                                                              sim_base_dir=self.sim_base_dir)
            return self.expectation_values
        else:
            assert False, "rel_results_path is undefined"

    def load_data_from_file(self, identifier="expectation_value",
                            custom_load_data_func=None, custom_load_data_args=None):
        from mcmctools.loading.loading import load_data_based_running_parameter
        data = load_data_based_running_parameter(
            rel_data_dir=self.rel_data_path, identifier=identifier, running_parameter=self.running_parameter,
            rp_values=self.rp_values,
            sim_base_dir=self.sim_base_dir, custom_load_data_func=custom_load_data_func,
            custom_load_data_args=custom_load_data_args)

        from mcmctools.utils.json import load_json_sim_params
        sim_params = load_json_sim_params(
            rel_data_path=self.rel_data_path, identifier=identifier,
            running_parameter=self.running_parameter, rp_key=self.rp_values[0], sim_base_dir=self.sim_base_dir)
        return data, sim_params

    @property
    def equilibrium_times(self):
        return self.computed_equilibrium_times

    @property
    def correlation_times(self):
        return self.computed_correlation_times

    @property
    def expectation_values(self):
        return self.computed_expectation_values