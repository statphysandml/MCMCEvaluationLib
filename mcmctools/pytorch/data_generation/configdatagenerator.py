import pandas as pd
import numpy as np

from pystatplottools.pytorch_data_generation.data_generation.datageneratorbaseclass import DataGeneratorBaseClass
from mcmctools.loading.loading import ConfigurationLoader


class ConfigDataGenerator(ConfigurationLoader, DataGeneratorBaseClass):

    def __init__(self, **kwargs):
        super().__init__(complex_number_format="plain", **kwargs)

        self.labels = kwargs.pop("labels", "running_parameter")  # Can also be referred to certain columns of data
        # self.complex_config_boolean = kwargs.pop("complex_config", False) # Needed?

        if self.labels == "running_parameter":
            self.labels = self.running_parameter.capitalize()

        if isinstance(self.labels, list):
            self.label_size = len(self.labels)
        else:
            self.label_size = 1

        self.data_type = kwargs.pop("data_type")
        if self.data_type == "target_config":
            self.sampler = self.sample_target_config
        elif self.data_type == "target_param":
            self.sampler = self.sample_target_param

        if self.skip_loading_data_in_init is False:
            # Load data (or first chunk)
            # Resamples the data in the chunk after loaded from file -> this does not necessarily resample from the
            # entire file, but only from the respective chunk
            self.data = self.get_next_chunk_collection(resample=True)
            self.determine_target_and_input_size()
            self.iterator = 0
        else:
            self.data = None
            # Results in a call of self.get_next_chunk_collection for the first sample that is drawn.
            self.iterator = self.chunksize * len(self.filenames)

    def determine_target_and_input_size(self):
        if self.data_type == "target_config":
            self.inp_size = self.label_size
            if hasattr(self.data["Config"].iloc[0], "__len__"):
                self.tar_size = len(self.data["Config"].iloc[0].values.flatten())
            else:
                self.tar_size = 1
        elif self.data_type == "target_param":
            if hasattr(self.data["Config"].iloc[0], "__len__"):
                self.inp_size = len(self.data["Config"].iloc[0].values.flatten())
            else:
                self.inp_size = 1
            self.tar_size = self.label_size

    def sample_target_config(self):
        if self.iterator == len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection(resample=True)  # load data
            # Needs to be set again if get_next_chunk_collection is called here for the first time
            self.determine_target_and_input_size()

        self.iterator += 1
        return np.array([self.data[self.labels].iloc[self.iterator - 1]]).reshape(self.inp_size), self.data["Config"].iloc[self.iterator - 1].values.reshape(self.tar_size)

    def sample_target_param(self):
        if self.iterator == len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection(resample=True)  # load data
            # Needs to be set again if get_next_chunk_collection is called here for the first time
            self.determine_target_and_input_size()

        self.iterator += 1
        return self.data["Config"].iloc[self.iterator - 1].values.reshape(self.inp_size), np.array([self.data[self.labels].iloc[self.iterator - 1]]).reshape(self.tar_size)
