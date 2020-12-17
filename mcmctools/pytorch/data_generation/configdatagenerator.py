import pandas as pd
import numpy as np

from pystatplottools.ppd_pytorch_data_generation.data_generation.datageneratorbaseclass import DataGeneratorBaseClass
from mcmctools.loading.loading import ConfigurationLoader

# Note that the class currently works only if the number of fixed parameters equals for all types of the same function


class ConfigDataGenerator(ConfigurationLoader, DataGeneratorBaseClass):

    def __init__(self, **kwargs):
        super().__init__(complex_number_format="plain", **kwargs)

        self.iterator = 0

        # Load data (or first chunk)
        # Resamples the data in the chunk after loaded from file -> this does not necessarily resample from the entire file, but only from the respective chunk
        self.data = self.get_next_chunk_collection(resample=True)

        self.labels = kwargs.pop("labels", "running_parameter")
        self.complex_config_boolean = kwargs.pop("complex_config", False)

        if self.labels == "running_parameter":
            self.labels = self.running_parameter.capitalize()

        if isinstance(self.labels, list):
            label_size = len(self.labels)
        else:
            label_size = 1

        self.data_type = kwargs.pop("data_type")
        if self.data_type == "target_config":
            self.inp_size = label_size
            self.tar_size = len(self.data["Config"].iloc[0].values.flatten())
            self.sampler = self.sample_target_config
        elif self.data_type == "target_param":
            self.inp_size = len(self.data["Config"].iloc[0].values.flatten())
            self.tar_size = label_size
            self.sampler = self.sample_target_param

        self.inspect = lambda axes, net, data_loader, device: self.inspect_magnetization(axes=axes, net=net,
                                                                                         data_loader=data_loader,
                                                                                         device=device)

    def sample_target_config(self):
        if self.iterator == len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection(resample=True)  # load data

        self.iterator += 1
        return np.array([self.data[self.labels].iloc[self.iterator - 1]]).reshape(self.inp_size), self.data["Config"].iloc[self.iterator - 1].values.reshape(self.tar_size)

    def sample_target_param(self):
        if self.iterator == len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection(resample=True)  # load data

        self.iterator += 1
        return self.data["Config"].iloc[self.iterator - 1].values.reshape(self.inp_size), np.array([self.data[self.labels].iloc[self.iterator - 1]]).reshape(self.tar_size)

    @staticmethod
    def retrieve_data(data_loader, device='cpu', net=None):
        beta_data = []
        config_data = []
        mean = []

        # from NetworkModels.inn import INN
        # if net is not None and isinstance(net, INN):
        #     import torch
        #     num = 400
        #     for i in range(100):
        #         betas = torch.tensor(np.transpose([np.repeat(np.linspace(0.1, 0.7, 25, dtype=np.float32), num)]))
        #         betas = betas.to(device)
        #         output = net.backward_step(betas)
        #         config_data += list(output[:, :16].detach().cpu().numpy())
        #         mean += list(np.mean(np.sign(output[:, :16].detach().cpu().numpy()), axis=1))
        #         # config_data += list(np.mean(output[:, :16].detach().cpu().numpy(), axis=1))
        #         beta_data += [beta[0].detach().cpu().numpy() for beta in betas]
        #
        #     # for batch_idx, (beta, config) in enumerate(data_loader):
        #     #     beta, config = beta.to(device), config.to(device)
        #     #
        #     #
        #     #     config_data.append(np.mean(output[0][:16].detach().cpu().numpy()))
        #     #     beta_data.append(beta[0].detach().cpu().numpy()[0])
        # else:
        for batch_idx, (betas, config) in enumerate(data_loader):
            config_data += [conf.detach().numpy() for conf in config]
            mean += [np.mean(conf.detach().numpy()) for conf in config]
            beta_data += [beta[0].detach().numpy() for beta in betas]

        config_data = np.array(config_data)
        beta_data = np.array(beta_data)
        mean = np.array(mean)
        beta_data = np.array([f"{bet:.3f}" for bet in beta_data])

        config_data = np.sign(config_data)

        absmean = np.abs(np.mean(np.sign(config_data), axis=1))
        # Todo: Add a computation of the Energy based on nearest neighbour

        data = pd.DataFrame(
            {"beta": beta_data, "Beta": np.array(beta_data, dtype=np.float32), "Mean": mean, "AbsMean": absmean})

        # Add AbsMean, Energy, and Higher Moments

        data.index.name = "Num"
        data.set_index(["beta"], append=True, inplace=True)
        data = data.reorder_levels(["beta", "Num"])
        data.sort_values(by="beta", inplace=True)
        return data, config_data

    # @staticmethod
    # def inspect_magnetization(axes, data_loader, device='cpu', net=None):
    #     from program.plot_routines.distribution1D import Distribution1D
    #     data, _ = ConfigDataGenerator.retrieve_data(data_loader=data_loader, device=device, net=net)
    #     dist1d = Distribution1D(data=data)
    #     dist1d.compute_histograms(columns=['Mean'], kind="probability_dist")
    #
    #     from program.plot_routines.contour2D import add_fancy_box
    #
    #     for i, idx in enumerate(data.index.unique(0)):
    #         histodat = dist1d.histograms[idx]['Mean']
    #         Distribution1D.plot_histogram(ax=axes[int(i * 1.0 / 7)][i % 7], **histodat)
    #         add_fancy_box(axes[int(i * 1.0 / 7)][i % 7], idx)
    #
    #     import matplotlib.pyplot as plt
    #     plt.tight_layout()

    # @staticmethod
    # def inspect_observables(axes, data_loader, device='cpu', net=None):
    #     # data, filenames = ConfigDataGenerator.load_all_configurations(
    #     #     "/remote/lin34/kades/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/HeatbathSmall")
    #     # data = pd.concat(data, keys=[f"{float(file[file.find('=') + 1:-4]):.3f}" for file in filenames])
    #     # data = data.sort_index(level=0)
    #
    #     data, config_data = ConfigDataGenerator.retrieve_data(data_loader=data_loader, device=device, net=net)
    #
    #     from program.plot_routines.distribution1D import Distribution1D
    #     dist1d = Distribution1D(data=data)
    #
    #     dist1d.compute_expectation_values(columns=['Mean', 'AbsMean'],  # , 'Energy'],
    #                                       exp_values=['mean', 'max', 'min', 'secondMoment', 'fourthMoment'])
    #     dist1d.compute_expectation_values(columns=['Beta'], exp_values=['mean'])
    #
    #     from program.plot_routines.distribution1D import compute_binder_cumulant
    #     # compute_specificheat(dist=dist1d, N=16)
    #     compute_binder_cumulant(dist=dist1d)
    #
    #     betas = dist1d.expectation_values['Beta']['mean']
    #
    #     axes[0][0].set_xlabel("$\\beta")
    #     axes[0][1].set_xlabel("$\\beta")
    #     axes[1][0].set_xlabel("$\\beta")
    #     axes[1][1].set_xlabel("$\\beta")
    #
    #     axes[0][0].set_ylabel("$\langle |m| \\rangle$")
    #     axes[0][1].set_ylabel("$c$")
    #     axes[1][0].set_ylabel("$U_L$")
    #     axes[1][1].set_ylabel("$\langle E \\rangle$")
    #
    #     axes[0][0].plot(betas, dist1d.expectation_values.loc[:, 'AbsMean']['mean'])
    #     axes[0][1].plot(betas, dist1d.expectation_values.loc[:, 'Mean']['mean'])
    #     # axes[0][1].plot(betas, dist1d.expectation_values.loc[:, 'SpecificHeat']['mean'].values)
    #     axes[1][0].plot(betas, dist1d.expectation_values.loc[:, 'BinderCumulant']['mean'])
    #     # axes[1][1].plot(betas, dist1d.expectation_values.loc[:, 'Energy']['mean'])
    #
    #     import matplotlib.pyplot as plt
    #     plt.tight_layout()


if __name__ == '__main__':
    pass

    ''' Newly commented'''

    # from program.data_generation.data_generation_base_classes.dataloaders import generate_data_loader
    # # from program.data_generation.ising_model.isingdatagenerator import ConfigDataGenerator
    #
    # n = 21 * 10000
    #
    # data_loader_params = {'batch_size': 1000,
    #     'shuffle': True,
    #     'num_workers': 0}
    #
    # from program.data_generation.data_generation_base_classes.datageneratorbaseclass import data_generator_factory
    # from program.data_generation.data_generation_base_classes.dataloaders import data_loader_factory
    #
    # data_loader_func = data_loader_factory(data_loader_name="BatchDataLoader")
    # data_generator_func = data_generator_factory(data_generator_name="BatchConfigDataGenerator")
    #
    # data_generator_args = {
    #     "data_type": "target_config",
    #     "path": "/home/lukas/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/IsingModel/",
    #     "chunksize": 10000,
    #     "total_number_of_data_per_file": 10000,
    #     "number_of_files": 21
    #     # "batch_size": 10
    # }
    #
    # data_loader = generate_data_loader(
    #     data_generator=data_generator_func,
    #     data_generator_args=data_generator_args,
    #     data_loader=data_loader_func,
    #     data_loader_params=data_loader_params,
    #     n=n,
    #     seed=0,
    #     device="cpu"
    # )
    #
    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots(3, 7, figsize=(14, 7))
    # ConfigDataGenerator.inspect_magnetization(axes=axes, data_loader=data_loader)
    # plt.show()
    #
    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots(2, 2)
    #
    # ConfigDataGenerator.inspect_observables(axes=axes, data_loader=data_loader)
    #
    # plt.show()

    ''' Example for computing and plotting observables '''

    # Computing observables

    # data, filenames = ConfigDataGenerator.load_all_configurations("/home/lukas/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/IsingModel/")
    # data = pd.concat(data, keys=[f"{float(file[file.find('=') + 1:-4]):.3f}" for file in filenames])
    # data = data.sort_index(level=0)
    #
    # from ising_model_data_loader.PlotRoutines.distribution1D import Distribution1D
    # dist1d = Distribution1D(data=data)
    #
    # dist1d.compute_expectation_values(columns=['Mean', 'AbsMean', 'Energy'],
    #                                   exp_values=['mean', 'max', 'min', 'secondMoment', 'fourthMoment'])
    # dist1d.compute_expectation_values(columns=['Beta'], exp_values=['mean'])
    #
    # from ising_model_data_loader.PlotRoutines.distribution1D import compute_specificheat, compute_binder_cumulant
    # compute_specificheat(dist=dist1d, N=16)
    # compute_binder_cumulant(dist=dist1d)
    #
    # from ising_model_data_loader.PlotRoutines.plotting_environment.loading_figure_mode import loading_figure_mode
    # fma = loading_figure_mode("saving")
    # import matplotlib.pyplot as plt
    # plt.style.use('seaborn-dark-palette')
    #
    # # Plot observables
    #
    # fig, axes = fma.newfig(1.7, nrows=2, ncols=2)
    #
    # betas = dist1d.expectation_values['Beta']['mean']
    #
    # axes[0][0].set_xlabel("$\\beta")
    # axes[0][1].set_xlabel("$\\beta")
    # axes[1][0].set_xlabel("$\\beta")
    # axes[1][1].set_xlabel("$\\beta")
    #
    # axes[0][0].set_ylabel("$\langle |m| \\rangle$")
    # axes[0][1].set_ylabel("$c$")
    # axes[1][0].set_ylabel("$U_L$")
    # axes[1][1].set_ylabel("$\langle E \\rangle$")
    #
    # axes[0][0].plot(betas, dist1d.expectation_values.loc[:, 'AbsMean']['mean'])
    # axes[0][1].plot(betas, dist1d.expectation_values.loc[:, 'SpecificHeat']['mean'].values)
    # axes[1][0].plot(betas, dist1d.expectation_values.loc[:, 'BinderCumulant']['mean'])
    # axes[1][1].plot(betas, dist1d.expectation_values.loc[:, 'Energy']['mean'])
    #
    # plt.tight_layout()
    #
    # fma.savefig("./../Examples/", "observables")
    #
    # # Example for computing and plotting histrograms for the resulting distribution per temper  ature for the magnetization
    #
    # dist1d.compute_histograms(columns=['Mean', 'AbsMean', 'Energy'], kind="probability_dist", nbins=10)
    #
    # fig, axes = fma.newfig(2.3, nrows=5, ncols=5, ratio=1)
    #
    # for i, idx in enumerate(data.index.unique(0).sort_values()):
    #     histodat = dist1d.histograms[idx]['Mean']
    #     axes[int(i * 1.0 / 5)][i % 5].set_xlim(-1, 1)
    #     Distribution1D.plot_histogram(ax=axes[int(i * 1.0 / 5)][i % 5], **histodat)
    #     from ising_model_data_loader.PlotRoutines.contour2D import add_fancy_box
    #     add_fancy_box(axes[int(i * 1.0 / 5)][i % 5], idx)
    #
    # plt.tight_layout()
    # fma.savefig("./../Examples/", "groundtruth")

    # from DataGeneration.datasetgenerator import generate_data_loader
    #
    # n = 2000*25
    #
    # data_generator_args = {
    #     "data_type": "target_config",
    #     "path": "/remote/lin34/kades/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/HeatbathSmall",
    #     "chunksize": 1,
    #     "total_number_of_data_per_file": 2000
    # }
    #
    # data_loader = generate_data_loader(
    #     data_generator=ConfigDataGenerator,
    #     data_generator_args=data_generator_args,
    #     data_loader_params={'batch_size': 1,
    #                         'shuffle': True,
    #                         'num_workers': 1},
    #     n=n,
    #     seed=0,
    #     set_seed=False)
    #
    # # import matplotlib.pyplot as plt
    # #
    # # fig, axes = plt.subplots(5, 5)
    #
    # # ConfigDataGenerator.inspect_magnetization(axes=axes, data_loader=data_loader)
    #
    # # plt.show()
    #
    # from plotting_environment.loading_figure_mode import loading_figure_mode
    #
    # fma = loading_figure_mode("saving")
    #
    # import matplotlib.pyplot as plt
    #
    # plt.style.use('seaborn-dark-palette')
    #
    # fig, axes = fma.newfig(3, nrows=5, ncols=5, ratio=1)
    #
    # ConfigDataGenerator.inspect_magnetization(axes=axes, data_loader=data_loader)
    #
    # fma.savefig(".", "test")
    #
    # import matplotlib.pyplot as plt
    #
    # plt.close()
