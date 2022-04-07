from pystatplottools.pdf_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=True)


import numpy as np


""" Example implementation for using the MCMCSimulation class with the Ising model defined in Python.

Providing the necessary methods, every kind of MCMC system can be simulated and evaluated by the methods and
modules of the MCMCEvaluationLib """


class IsingModel:
    def __init__(self, beta, J, h, dimensions, measures=[]):
        # Required for mode simulation
        self.beta = beta
        self.J = J
        self.h = h
        self.dimensions = dimensions
        self._measures = measures

        self.n_sites = np.product(self.dimensions)

        self.lattice = None
        self.neighbour_indices = None

        self._precompute_neighbour_indices()

    def initialize(self, starting_mode):
        if starting_mode == "hot":
            self.lattice = 2 * np.random.randint(0, 2, self.n_sites, dtype=np.int8) - 1
        else:
            self.lattice = np.ones(self.n_sites, dtype=np.int8)

    @property
    def measure_names(self):
        return self._measures

    @measure_names.setter
    def measure_names(self, measures):
        self._measures = measures

    def update(self, n_step):
        random_sites = np.random.randint(0, self.n_sites, n_step)
        rand = np.random.rand(n_step)
        for step, rnd_site in enumerate(random_sites):
            if rand[step] < np.exp(-2.0 * self.beta * self.lattice[rnd_site] *
                                   (self.J * np.sum(self.lattice[self.neighbour_indices[rnd_site]]) + self.h)):
                self.lattice[rnd_site] *= -1

    # def measure(self):
    #     # Can be used for measurement_to_file in mcmc_simulation.py
    #     measures = ""
    #     for measure in self.measure_names:
    #         if measure == "Mean":
    #             measures += f'{self.lattice.mean():.6f}' + "\t"
    #     return measures[:-1]

    def measure(self):
        measures = []

        for measure in self.measure_names:
            if measure == "Mean":
                measures.append(self.lattice.mean())
            elif measure == "AbsMean":
                measures.append(np.abs(self.lattice.mean()))
            elif measure == "Config":
                config = ""
                for site in self.lattice:
                    config += f'{site:.6f}' + " "
                measures.append(config[:-1])
            else:
                assert False, "Unkown measure" + measure
        return measures

    def _precompute_neighbour_indices(self):
        from mcmctools.utils.lattice import get_neighbour_index
        dim_mul = np.cumprod([1] + list(self.dimensions))

        self.neighbour_indices = np.zeros((self.n_sites, 2 * len(self.dimensions)), np.int32)
        for site in range(self.n_sites):
            for dim in range(len(self.dimensions)):
                self.neighbour_indices[site, 2 * dim] = get_neighbour_index(
                    n=site, dim=dim, direction=True, mu=0, dimensions=self.dimensions, dim_mul=dim_mul, elem_per_site=1)
                self.neighbour_indices[site, 2 * dim + 1] = get_neighbour_index(
                    n=site, dim=dim, direction=False, mu=0, dimensions=self.dimensions, dim_mul=dim_mul, elem_per_site=1)


if __name__ == "__main__":
    ising_model = IsingModel(beta=0.1, J=1.0, h=0.0, dimensions=[4, 4])

    running_parameter = "beta"
    rp_values = [0.1, 0.4, 0.6]

    from mcmc.mcmc_simulation import MCMCSimulation
    simulation = MCMCSimulation(model=ising_model,
                                running_parameter=running_parameter,
                                rp_values=rp_values)

    # Expectation value simulation and evaluation
    simulation.run_expectation_value_simulation(measures=["Mean", "Config"], n_measurements=1000,
                                                n_steps_equilibrium=100, n_steps_autocorrelation=10,
                                                starting_mode="hot")
    data = simulation.measurements_to_dataframe()

    from mcmctools.modes.expectation_value import expectation_value
    expectation_values = expectation_value(
        number_of_measurements=1000, measures=["Mean", "AbsMean"],
        error_type="statistical",
        running_parameter=running_parameter, rp_values=rp_values,
        data=data)
    print(expectation_values)