import pandas as pd
import numpy as np
import os
import glob
import math


from pystatplottools.utils.multiple_inheritance_base_class import MHBC


class ConfigurationLoader(MHBC):
    def __init__(self, **kwargs):
        # For child classes with more than one parent class
        super().__init__(**kwargs)

        self.path = kwargs.pop('path')
        self.total_number_of_data_per_file = kwargs.pop("total_number_of_data_per_file")
        self.identifier = kwargs.pop("identifier", "")
        self.running_parameter = kwargs.pop("running_parameter", "default")
        self.drop_last = kwargs.pop("drop_last", False)  # Should only be used if it is wanted that each chunk has the same length
        self.complex_number_format = kwargs.pop("complex_number_format", "complex")  # or "plain"
        self.skipcols = kwargs.pop("skipcols", None)
        self.transformer = kwargs.pop("transformer", None)  # For passing functions
        self.transform = kwargs.pop("transform", False)  # Looks for transformer function in raw/transformer.py
        self.transformer_path = kwargs.pop("transformer_path", self.path + "/raw")
        if self.transformer_path is not None:
            import sys
            sys.path.append(os.path.abspath(self.transformer_path))

        # Chunksize and total number of chunks
        self.chunksize = kwargs.pop("chunksize", self.total_number_of_data_per_file)
        if self.drop_last:
            # Not working for resulting self.total_chunks = 1 since than all data is loaded at a later point
            self.total_chunks = math.floor(self.total_number_of_data_per_file * 1.0 / self.chunksize)
        else:
            self.total_chunks = math.ceil(self.total_number_of_data_per_file * 1.0 / self.chunksize)

        self.chunk_iterator = 0
        self.skiprows = 0

        # Enables to continue read the file from a given chunk iterator val -> not sure what it does , so far
        current_chunk_iterator_val = kwargs.pop("current_chunk_iterator_val", None)
        if current_chunk_iterator_val is not None:
            self.skiprows = range(1, current_chunk_iterator_val * self.chunksize + 1)
            self.chunk_iterator = current_chunk_iterator_val

        # Assign a reader to each file
        self.readers, self.filenames = ConfigurationLoader.load_configuration_readers(
            path=self.path,
            nrows=self.total_number_of_data_per_file,
            chunksize=self.chunksize,
            skiprows=self.skiprows,
            identifier=self.identifier,
            skipcols=self.skipcols
        )

        if self.total_chunks == 1:
            # Used to store the data if only a single chunk is loaded
            self.data = None

        self.total_number_of_repeated_loading = 0

    def __len__(self):
        return self.total_number_of_data_per_file * len(self.filenames)

    def get_next_chunk_collection(self, resample=True):
        if self.total_chunks == 1:
            self.total_number_of_repeated_loading += 1
            return self.load_all_data(resample)

        if self.chunk_iterator >= self.total_chunks:
            self.total_number_of_repeated_loading += 1
            # Readers and filenames are reloaded for the next iteration
            self.readers, self.filenames = ConfigurationLoader.load_configuration_readers(
                path=self.path,
                nrows=self.total_number_of_data_per_file,
                chunksize=self.chunksize,
                identifier=self.identifier,
                skipcols=self.skipcols
            )
            self.chunk_iterator = 0

        data = []
        for idx, reader in enumerate(self.readers):
            chunk = next(reader)
            chunk = ConfigurationLoader.prepare_single_data_frame(
                dat=chunk, file=self.filenames[idx], running_parameter=self.running_parameter)
            data.append(chunk)

        data = ConfigurationLoader.merge_file_datastreams(data=data, resample=resample)
        data = ConfigurationLoader.transform_config_data(data=data, complex_number_format=self.complex_number_format)

        if self.transform:
            try:
                from raw_transformer import transformer
            except ModuleNotFoundError:
                import sys
                sys.exit("ModuleNotFoundError: raw_transformer.py module not found. Needs to be set via transformer"
                         "path or by adding path of raw_transformer.py to sys.path")
            data = transformer(data)
        elif self.transformer is not None:
            data = self.transformer(data)

        if self.running_parameter == "default":
            del data["Default"]

        self.chunk_iterator += 1

        return data

    def get_chunk_iterator(self):
        return self.chunk_iterator

    def load_all_data(self, resample=True):
        if self.data is None:
            # Load all data
            self.data = self.load_all_data_ordered_by_index()
            self.data = self.data.droplevel(0).reset_index(drop=True)
            self.chunk_iterator += 1
        if resample:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        return self.data

    def load_all_data_ordered_by_index(self):
        return ConfigurationLoader.load_all_configurations(
            path=self.path,
            identifier=self.identifier,
            running_parameter=self.running_parameter,
            nrows=self.total_number_of_data_per_file,
            complex_number_format=self.complex_number_format,
            transform=self.transform,
            transformer=self.transformer,
            transformer_path=None  # Already added to sys.path in init function
        )[0]
    
    @staticmethod
    def load_configuration_readers(path, nrows, chunksize=100, skiprows=0, identifier="", skipcols=None):
        readers = []
        filenames = []

        current_directory = os.path.abspath(os.getcwd())

        os.chdir(path)
        for file in glob.glob(identifier + "*.dat"):
            if skipcols is not None:
                with open(file) as f:
                    header_line = f.readline()
                usecols = [item for item in header_line.split("\t") if item not in skipcols]
            else:
                usecols = None

            reader = pd.read_csv(file, delimiter="\t", header=0, chunksize=chunksize, skiprows=skiprows, index_col=False, nrows=nrows, usecols=usecols)
            readers.append(reader)
            filenames.append(file)

        os.chdir(current_directory)
        return readers, filenames  # , chunk_order

    @staticmethod
    def load_all_configurations(path, identifier="None", running_parameter="default", skiprows=0, nrows="all", complex_number_format="complex", skipcols=None, transformer=None, transform=False, transformer_path=None):
        data = []
        filenames = []

        current_directory = os.path.abspath(os.getcwd())

        if transformer_path is not None:
            import sys
            sys.path.append(os.path.abspath(transformer_path + "/raw"))

        os.chdir(path)

        data_files = glob.glob(identifier + "*.dat")

        for file in data_files:
            if skipcols is not None:
                with open(file) as f:
                    header_line = f.readline()
                header_line = header_line[:-1]
                usecols = [item for item in header_line.split("\t") if item not in skipcols]
            else:
                usecols = None

            if nrows == "all":
                dat = pd.read_csv(file, delimiter="\t", header=0, skiprows=skiprows, index_col=False, usecols=usecols)
            else:
                dat = pd.read_csv(file, delimiter="\t", header=0, skiprows=skiprows, index_col=False, nrows=nrows, usecols=usecols)
            dat = ConfigurationLoader.prepare_single_data_frame(dat=dat, file=file, running_parameter=running_parameter)

            data.append(dat)
            filenames.append(file)

        data = ConfigurationLoader.merge_file_datastreams_by_index(data=data, by_col_index=running_parameter)
        data = ConfigurationLoader.transform_config_data(data=data, complex_number_format=complex_number_format)

        if transform:
            try:
                from raw_transformer import transformer
            except ModuleNotFoundError:
                import sys
                sys.exit("ModuleNotFoundError: raw_transformer.py module not found. Needs to be set via transformer"
                         "path or by adding path of raw_transformer.py to sys.path")
            data = transformer(data)
        elif transformer is not None:
            data = transformer(data)

        if running_parameter == "default":
            del data["Default"]
        os.chdir(current_directory)

        return data, filenames  # , chunk_order

    @staticmethod
    def prepare_single_data_frame(dat, file, running_parameter):
        if "Unnamed" in dat.columns[-1]:
            dat.drop(dat.columns[len(dat.columns) - 1], axis=1, inplace=True)

        #  Multiple data files are loaded based on running parameter
        if running_parameter != "default":
            running_parameter_val = np.float32(file[file.find("=") + 1:file.find(".dat")])
        # Load single data file -> only a single file is loaded
        else:
            running_parameter_val = running_parameter

        dat = dat.assign(**{running_parameter.capitalize(): running_parameter_val})
        return dat

    # Plain merge -> single-index data frame of the samples
    @staticmethod
    def merge_file_datastreams(data, resample=False):
        data = pd.concat(data)
        data = data.reset_index(drop=True)
        if resample:
            data = data.sample(frac=1).reset_index(drop=True)
        return data

    # Keeps by_col_index as upper level index -> multi-index data frame of the samples
    @staticmethod
    def merge_file_datastreams_by_index(data, by_col_index=None):
        # Loaded only one file - adds default as an index
        if isinstance(data[0].loc[0, by_col_index.capitalize()], str):
            keys = [data[0].loc[0, by_col_index.capitalize()]]
        # Loaded several files
        else:
            keys = [f"{x.loc[0, by_col_index.capitalize()]:.6f}" for x in data]
        data = pd.concat(data, keys=keys).sort_index(level=0)
        data.index.set_names([by_col_index, 'sample_num'], inplace=True)
        return data

    @staticmethod
    def transform_config_data(data, complex_number_format="complex"):
        if "Config" in data and data["Config"].dtype == object:
            if data["Config"].iloc[0].find(" i") != -1:
                complex_num = True
            else:
                complex_num = False

            # Determine the number of sites
            n_sites = len(data["Config"].iloc[0].split(", "))

            # Split everything
            data["Config"] = data["Config"].apply(
                lambda x: np.float32(x.replace(" i", ", ").replace(", ", " ").split(" ")))

            if complex_num and complex_number_format == "complex":
                # Combine to a complex number
                data["Config"] = data["Config"].apply(lambda x: x[::2] + 1.0j * x[1::2])

            # Extract entry from array with single element
            if len(data["Config"].iloc[0]) == 1:
                # Single-index
                data["Config"] = data["Config"].apply(lambda x: x[0])
            elif len(data["Config"].iloc[0]) == n_sites:
                # Multi-index with config-elements
                column_index = pd.MultiIndex.from_product([["Config"], range(0, n_sites)],
                                                          names=["quantity", "elem"])
                row_index = data.index
                configs_multi_index = pd.DataFrame(data=np.stack(data["Config"].values, axis=0), index=row_index,
                                                   columns=column_index)

                data = data.drop("Config", axis=1)
                data.columns = pd.MultiIndex.from_tuples([(c, '') for c in data],
                                                         names=["quantity", "elem"])
                data = pd.concat([data, configs_multi_index], axis=1)
            else:
                # Multi-index with config-elements and components
                column_index = pd.MultiIndex.from_product([["Config"], range(0, n_sites), range(0, int(len(data["Config"].iloc[0]) / n_sites))],
                                                          names=["quantity", "elem", "component"])
                row_index = data.index
                configs_multi_index = pd.DataFrame(data=np.stack(data["Config"].values, axis=0), index=row_index,
                                                   columns=column_index)

                data = data.drop("Config", axis=1)
                data.columns = pd.MultiIndex.from_tuples([(c, '', '') for c in data],
                                                         names=["quantity", "elem", "component"])
                data = pd.concat([data, configs_multi_index], axis=1)

        else:
            # Determine data column index in dependence of entry types
            for col in data.columns:
                if data[col].dtype != object or col == "Config" or col == "Default" or "Default" in col:
                    continue

                # Determine number of components
                x = data[col].iloc[0]
                if x.find(" i") != -1:
                    complex_num = True
                else:
                    complex_num = False

                x = np.float32(x.replace(" i", ", ").replace(", ", " ").split(" "))
                if complex_num and complex_number_format == "complex":
                    x = x[::2] + 1.0j * x[1::2]
                if len(x) != 1:
                    data.columns = pd.MultiIndex.from_tuples([(c, '') for c in data],
                                                             names=["quantity", "component"])

        for col in data.columns:
            if data[col].dtype != object or col == "Config" or col == "Default" or "Default" in col:
                continue

            if data[col].iloc[0].find(" i") != -1:
                complex_num = True
            else:
                complex_num = False

            # Split everything
            data[col] = data[col].apply(lambda x: np.float32(x.replace(" i", ", ").replace(", ", " ").split(" ")))

            if complex_num and complex_number_format == "complex":
                # Combine to a complex number
                data[col] = data[col].apply(lambda x: x[::2] + 1.0j * x[1::2])

            # Extract entry from array with single element
            if len(data[col].iloc[0]) == 1:
                # Single-index data frame
                data[col] = data[col].apply(lambda x: x[0])
            else:
                if data.columns.nlevels == 2:
                    # Multi-index data frame with components
                    column_index = pd.MultiIndex.from_product([[col[0]], range(0, len(data[col].iloc[0]))],
                                                              names=["quantity", "component"])
                else:
                    # Multi-index data frame with config-elements and components
                    column_index = pd.MultiIndex.from_product([[col[0]], [""], range(0, len(data[col].iloc[0]))],
                                                              names=["quantity", "elem", "component"])

                row_index = data.index
                col_multi_index = pd.DataFrame(data=np.stack(data[col].values, axis=0), index=row_index,
                                               columns=column_index)
                data = data.drop(col, axis=1)
                data = pd.concat([data, col_multi_index], axis=1)

        return data


def load_data(files_dir, running_parameter, identifier, skipcols=None, complex_number_format="complex"):
    data_path = os.getcwd() + "/data/" + files_dir

    data, filenames = ConfigurationLoader.load_all_configurations(
        path=data_path,
        identifier=identifier,
        running_parameter=running_parameter,
        skipcols=skipcols,
        complex_number_format=complex_number_format
    )
    return data, filenames
