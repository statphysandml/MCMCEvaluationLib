import pandas as pd


from mcmctools.mcmc.evaluation_module import EvaluationModule


class Model:
    def __init__(self, measures):
        self.measure_names = measures
        pass

    def initialize(self, starting_mode):
        pass

    @property
    def measure_names(self):
        return self.__measures

    @measure_names.setter
    def measure_names(self, measures):
        self.__measures = measures

    def update(self, n_step):
        pass

    def measure(self):
        pass


class MCMCSimulation(EvaluationModule):
    def __init__(self, model, default_measures=[], sim_base_dir=None,
                 rel_data_path=None,  # -> sim_base_dir + "/" + rel_data_path
                 rel_results_path=None,  # -> sim_base_dir + "/" + rel_results_path
                 running_parameter_kind=None,
                 running_parameter=None,
                 rp_values=None):
        super().__init__(sim_base_dir=sim_base_dir, rel_data_path=rel_data_path, rel_results_path=rel_results_path,
                         running_parameter_kind=running_parameter_kind, running_parameter=running_parameter,
                         rp_values=rp_values)

        self.model = model
        self.measurements = {}

    def initialize_model(self, rp_val=None, starting_mode="hot"):
        if rp_val is not None:
            setattr(self.model, self.running_parameter, rp_val)

        self.model.initialize(starting_mode=starting_mode)

    def update(self, n_steps):
        self.model.update(n_steps)

    def measure(self):
        measurements = self.model.measure()
        rp_val = getattr(self.model, self.running_parameter)
        for measurement, measure_name in zip(measurements, self.model.measure_names):
            self.measurements[rp_val][measure_name].append(measurement)

    def initialize_measurements(self, measures):
        self.model.measure_names = measures
        self.measurements = {rp_val: {measure: [] for measure in self.model.measure_names} for rp_val in self.rp_values}

    def measurements_to_dataframe(self, complex_number_format="complex", transformer=None, transform=False,
                                  transformer_path=None):
        from mcmctools.loading.loading import ConfigurationLoader
        n_measurements = len(self.measurements[self.rp_values[0]][self.model.measure_names[0]])
        data = ConfigurationLoader.process_mcmc_configurations(
            data=[pd.DataFrame({**item, self.running_parameter.capitalize(): [key] * n_measurements}) for key, item in
                  self.measurements.items()],
            running_parameter=self.running_parameter, complex_number_format=complex_number_format,
            transformer=transformer, transform=transform, transformer_path=transformer_path)
        return data

    def measurements_to_file(self):
        pass

    def run_equilibrium_time_simulation(self, measure, sample_size, number_of_steps):
        self.initialize_measurements(measures=[measure])

        # Possibility to define a custom __iter__ class - or several for each mode...
        for rp_val in self.rp_values:
            starting_mode = "hot"
            for m in range(2 * sample_size):
                self.initialize_model(starting_mode=starting_mode, rp_val=rp_val)
                self.measure()
                for n in range(number_of_steps - 1):
                    self.update(n_steps=1)
                    self.measure()

                if starting_mode == "hot":
                    starting_mode = "cold"
                else:
                    starting_mode = "hot"

    def run_correlation_time_simulation(self, measure, minimum_sample_size, maximum_correlation_time, start_measuring):
        self.initialize_measurements(measures=[measure])

        # Possibility to define a custom __iter__ class - or several for each mode...
        for rp_val in self.rp_values:
            for m in range(minimum_sample_size):
                self.initialize_model(starting_mode="hot", rp_val=rp_val)
                self.update(n_steps=start_measuring)
                self.measure()
                for n in range(maximum_correlation_time - 1):
                    self.update(n_steps=1)
                    self.measure()

    def run_expectation_value_simulation(self, measures, n_measurements, n_steps_equilibrium, n_steps_autocorrelation,
                                         starting_mode="hot"):
        self.initialize_measurements(measures=measures)

        # Possibility to define a custom __iter__ class - or several for each mode...
        for rp_val in self.rp_values:
            self.initialize_model(starting_mode=starting_mode, rp_val=rp_val)
            self.update(n_steps=n_steps_equilibrium)
            self.measure()
            for n in range(n_measurements - 1):
                self.update(n_steps=n_steps_autocorrelation)
                self.measure()
