
from mcmctools.pytorch.data_generation.configdatagenerator import ConfigDataGenerator

# Avoids that samples are loaded one by another from the self.data dataframe - instead, a batch is extracted directly
# This leads to a performance boost since the underlying data frame is accessed via slicing
# i.e: batch = self.data.iloc[i:i+batch_size].values instead of
#    : batch = np.stack([self.data.iloc[j] for j in range(i, i+batch_size])


class BatchConfigDataGenerator(ConfigDataGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = kwargs.pop('batch_size')

    def sample_target_config(self):
        if self.iterator >= len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection(resample=True)  # load data

        self.iterator += self.batch_size

        if self.iterator > len(self.data):
            batch, target = list(self.data[self.labels].iloc[self.iterator - self.batch_size:len(self.data)].values.reshape((-1, self.inp_size))) , \
                            list(self.data["Config"].iloc[self.iterator - self.batch_size:len(self.data)].values.reshape((-1, self.tar_size)))

            if self.chunk_iterator < self.total_chunks:
                # Load next chunk and reset iterator
                n_missing_configs = self.iterator - len(self.data)
                self.iterator = n_missing_configs  # Reset iterator
                self.data = self.get_next_chunk_collection(resample=True)  # load data
                batch += list(self.data[self.labels].iloc[0:n_missing_configs].values.reshape((-1, self.inp_size)))
                target += list(self.data["Config"].iloc[0:n_missing_configs].values.reshape((-1, self.tar_size)))
                return batch, target
            else:
                # End of files has been reached
                # Prepare next data iteration
                self.iterator = 0  # Reset iterator
                self.data = self.get_next_chunk_collection(resample=True)  # load data
                # Return last samples of previous data iteration
                return batch, target
        else:
            return list(self.data[self.labels].iloc[self.iterator - self.batch_size:self.iterator].values.reshape((-1, self.inp_size))) , \
                   list(self.data["Config"].iloc[self.iterator - self.batch_size:self.iterator].values.reshape((-1, self.tar_size)))

    def sample_target_param(self):
        if self.iterator == len(self.data):
            # Load next chunk and reset iterator
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection(resample=True)  # load data

        self.iterator += self.batch_size

        if self.iterator > len(self.data):
            batch, target = list(self.data["Config"].iloc[self.iterator - self.batch_size:len(self.data)].values.reshape(-1, self.inp_size)), \
                            list(self.data[self.labels].iloc[self.iterator - self.batch_size:len(self.data)].values.reshape((-1, self.tar_size)))

            if self.chunk_iterator < self.total_chunks:
                # Load next chunk and reset iterator
                n_missing_configs = self.iterator - len(self.data)
                self.iterator = n_missing_configs  # Reset iterator
                self.data = self.get_next_chunk_collection(resample=True)  # load data
                batch += list(self.data["Config"].iloc[0:n_missing_configs].values.reshape(-1, self.inp_size))
                target += list(self.data[self.labels].iloc[0:n_missing_configs].values.reshape((-1, self.tar_size)))
                return batch, target
            else:
                # End of files has been reached
                # Prepare next data iteration
                self.iterator = 0  # Reset iterator
                self.data = self.get_next_chunk_collection(resample=True)  # load data
                # Return last samples of previous data iteration
                return batch, target
        else:
            return list(self.data["Config"].iloc[self.iterator - self.batch_size:self.iterator].values.reshape(-1, self.inp_size)), \
                   list(self.data[self.labels].iloc[self.iterator - self.batch_size:self.iterator].values.reshape((-1, self.tar_size)))