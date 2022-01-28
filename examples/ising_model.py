from pystatplottools.pdf_env.loading_figure_mode import loading_figure_mode

fma, plt = loading_figure_mode(develop=False)

from mcmc.mcmc_simulation import MCMCSimulation
from mcmctools.utils.lattice import get_neighbour_index

import numpy as np


class IsingModel:
    def __init__(self, beta, J, h, dimensions, measures=[]):
        # Required for mode simulation
        self.beta = beta
        self.J = J
        self.h = h
        self.dimensions = dimensions
        self.measures = measures

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
        return self.measures

    @measure_names.setter
    def measure_names(self, measures):
        self.measures = measures

    def update(self, n_step):
        random_sites = np.random.randint(0, self.n_sites, n_step)
        rand = np.random.rand(n_step)
        for step in range(n_step):
            if rand[step] < np.exp(-self.beta * (self.J * np.sum(self.lattice[self.neighbour_indices[random_sites]], axis=1) + self.h)):
                self.lattice[random_sites] *= -1

    # def measure(self):
    #     # Can be used for measurement_to_file in mcmc_simulation.py
    #     measures = ""
    #     for measure in self.measures:
    #         if measure == "Mean":
    #             measures += f'{self.lattice.mean():.6f}' + "\t"
    #     return measures[:-1]

    def measure(self):
        measures = []

        for measure in self.measures:
            if measure == "Mean":
                measures.append(self.lattice.mean())
            else:
                assert False, "Unkown measure" + measure
        return measures

    def _precompute_neighbour_indices(self):
        dim_mul = np.cumprod([1] + list(self.dimensions))

        self.neighbour_indices = np.zeros((self.n_sites, 2 * len(self.dimensions)), np.int)
        for site in range(self.n_sites):
            for dim in range(len(self.dimensions)):
                self.neighbour_indices[site, 2 * dim] = get_neighbour_index(
                    n=site, dim=dim, direction=True, mu=0, dimensions=self.dimensions, dim_mul=dim_mul, elem_per_site=1)
                self.neighbour_indices[site, 2 * dim + 1] = get_neighbour_index(
                    n=site, dim=dim, direction=False, mu=0, dimensions=self.dimensions, dim_mul=dim_mul, elem_per_site=1)


if __name__ == "__main__":
    ising_model = IsingModel(beta=0.1, J=1.0, h=0.0, dimensions=[4, 4])

    simulation = MCMCSimulation(model=ising_model,
                                rel_data_path="./data/Test/",
                                rel_results_path="./data/Test/results/",
                                # running_parameter_kind="model_params",
                                running_parameter="beta",
                                rp_values=[0.1, 0.4, 0.7])

    simulation.run_equilibrium_time_simulation(measure="Mean", sample_size=10, number_of_steps=100)

    data = simulation.measurements_to_dataframe()
    simulation.compute_equilibrium_time(data=data, sample_size=10, number_of_steps=100, eval_confidence_range=0.1,
                                        eval_confidence_window=10, measure="Mean", fma=fma)