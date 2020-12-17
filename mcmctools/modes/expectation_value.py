import sys
import os
import copy
import numpy as np
import pandas as pd

from pystatplottools.ppd_distributions.expectation_value import ExpectationValue

from mcmctools.utils.json import load_configs
from mcmctools.loading.loading import load_data


def compute_measure_over_config(data, measure_name, sim_params):
    if measure_name == "2ndMoment":
        return compute_nth_moment(data, 2, measure_name)
    elif measure_name == "3rdMoment":
        return compute_nth_moment(data, 3, measure_name)
    elif measure_name == "AbsMean":
        return compute_abs_mean(data)


def compute_mean(data):
    if data.columns.nlevels == 1:
        # Mean equal to single config
        data.insert(len(data.columns), "Mean", data["Config"])
    elif data.columns.nlevels == 2:
        # Mean over multiple elements
        data.insert(len(data.columns), "Mean", data["Config"].to_numpy().reshape(len(data), -1).mean(axis=1))
    else:
        n_components = len(data["Config", 0].iloc[0])
        mean = data["Config"].to_numpy().reshape(len(data), -1, n_components).mean(axis=1)
        row_index = data["Config"].index
        column_index = pd.MultiIndex.from_product([["Mean"], [""], range(0, n_components)],
                                                  names=["quantity", "elem", "component"])
        data = pd.concat([data, pd.DataFrame(data=mean, index=row_index, columns=column_index)], axis=1)
    return ["Mean"], data


def compute_nth_moment(data, n, measure_name):
    new_measures = []

    if "Mean" not in data.columns.get_level_values(0):
        new_mean_measure, data = compute_mean(data)
        new_measures += new_mean_measure

    if data.columns.nlevels <= 2:
        data.insert(len(data.columns), measure_name, data["Mean"].apply(lambda x: np.power(x, n)))
    else:
        n_components = len(data["Mean"].iloc[0])
        column_index = pd.MultiIndex.from_product([[measure_name], [""], range(0, n_components)],
                                                  names=["quantity", "elem", "component"])
        moment = data["Mean"].apply(lambda x: np.power(x, n)).values
        data = pd.concat([data, pd.DataFrame(data=moment, index=data["Mean"].index, columns=column_index)], axis=1)

    new_measures.append(measure_name)

    return new_measures, data


def compute_abs_mean(data):
    data.insert(len(data.columns), "AbsMean", data["Config"].apply(lambda x: np.abs(np.mean(x))))
    return ["AbsMean"], data


def compute_measures_over_config(data, measures, sim_params):
    effective_measures = copy.deepcopy(measures)
    for measure in measures:
        if measure not in data.columns:
            new_measures, data = compute_measure_over_config(data=data, measure_name=measure, sim_params=sim_params)
            if len(new_measures) > 0:
                effective_measures.remove(measure)
                effective_measures += new_measures
    return effective_measures, data


def expectation_value(files_dir, sim_root_dir="", rel_path="./"):
    # Load configs and data
    cwd = os.getcwd()

    sim_params, execution_params, running_parameter = load_configs(files_dir=files_dir, mode="expectation_value")
    data, filenames = load_data(files_dir=files_dir, running_parameter=running_parameter, identifier="expectation_value")

    # Compute measures based on the given configurations that have not been computed during the simulation
    post_measures = execution_params["post_measures"]
    if post_measures is not None:
        post_measures, data = compute_measures_over_config(data=data, measures=post_measures, sim_params=sim_params)

    # Compute the expectation values and the error
    ep = ExpectationValue(data=data)

    expectation_value_measures = []
    if sim_params["systembase_params"]["measures"] is not None:
        expectation_value_measures += sim_params["systembase_params"]["measures"]
    if post_measures is not None:
        expectation_value_measures += post_measures

    black_expectation_value_list = ["Config"]
    expectation_value_measures = [
        exp_value for exp_value in expectation_value_measures if exp_value not in black_expectation_value_list]

    ep.compute_expectation_value(columns=expectation_value_measures,
                                 exp_values=['mean'])  # , 'max', 'min' , 'secondMoment', 'fourthMoment'
    expectation_values = ep.expectation_values

    if "n_means_bootstrap" in execution_params.keys() and execution_params["n_means_bootstrap"] != 0:
        ep.compute_error_with_bootstrap(n_means_boostrap=execution_params["n_means_bootstrap"],
                                        number_of_measurements=execution_params["number_of_measurements"],
                                        columns=expectation_value_measures,
                                        exp_values=['mean'],
                                        running_parameter=running_parameter)
    else:
        ep.compute_std_error(columns=expectation_value_measures)

    errors = ep.errors

    expectation_values = ep.drop_multiindex_levels_with_unique_entries(data=expectation_values)
    errors = ep.drop_multiindex_levels_with_unique_entries(data=errors)
    results = pd.concat([expectation_values, errors], keys=["ExpVal", "Error"], axis=1)

    results = results.transpose()
    results = results.sort_index(level=1, sort_remaining=False)
    results = results.reset_index()
    results = results.transpose()
    results = results.reset_index()

    if not os.path.isdir(os.getcwd() + "/results/" + files_dir):
        os.makedirs(os.getcwd() + "/results/" + files_dir)

    results = results.applymap(str)
    results.to_json(os.getcwd() + "/results/" + files_dir + "/expectation_value_results.json", indent=4)

    os.chdir(cwd)


def load_expectation_value_results(files_dir):
    if os.path.exists(os.getcwd() + "/results/" + files_dir + "/expectation_value_results.json"):
        results = pd.read_json(os.getcwd() + "/results/" + files_dir + "/expectation_value_results.json",
                               convert_dates=False,  # dont convert columns to dates
                               convert_axes=False  # dont convert index to dates
                               )
        sim_params, execution_params, running_parameter = load_configs(files_dir=files_dir, mode="expectation_value")
        results = results.set_index(running_parameter)

        column_levels = []
        for item_idx in results.index.values:
            if "level" in item_idx:
                column_levels.append(item_idx)
            else:
                break
        results = results.transpose()
        results = results.set_index(column_levels)
        results = results.transpose()
        return results
    else:
        return None


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("FilesDir:", sys.argv[1])  # , "SimRootDir:", sys.argv[2], "RelPath:", sys.argv[3])
        expectation_value(sys.argv[1])  # , sys.argv[2], sys.argv[3])
    else:
        os.chdir("../../../examples/")
        expectation_value(files_dir="IsingModelSimulation")
        load_expectation_value_results(files_dir="IsingModelSimulation")



