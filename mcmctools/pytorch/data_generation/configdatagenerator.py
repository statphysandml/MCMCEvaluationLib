import pandas as pd
import numpy as np

from pystatplottools.pytorch_data_generation.data_generation.datageneratorbaseclass import DataGeneratorBaseClass
from mcmctools.loading.loading import ConfigurationLoader

# Note that the class currently works only if the number of fixed parameters equals for all types of the same function


class ConfigDataGenerator(ConfigurationLoader, DataGeneratorBaseClass):

    def __init__(self, **kwargs):
        super().__init__(complex_number_format="plain", **kwargs)

        self.iterator = 0

        # Load data (or first chunk)
        # Resamples the data in the chunk after loaded from file -> this does not necessarily resample from the entire
        # file, but only from the respective chunk
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
