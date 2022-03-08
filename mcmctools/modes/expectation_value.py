import sys
import os
import copy
import numpy as np
import pandas as pd


from traceback import print_exc


from pystatplottools.expectation_values.expectation_value import ExpectationValue

from mcmctools.utils.utils import to_float
from mcmctools.loading.loading import load_data_based_running_parameter


def compute_measure_over_config(data, measure_name, custom_measures_func=None, custom_measures_args=None):
    if measure_name == "SecondMoment":
        return compute_nth_moment(data, 2, measure_name)
    elif measure_name == "ThirdMoment":
        return compute_nth_moment(data, 3, measure_name)
    elif measure_name == "FourthMoment":
        return compute_nth_moment(data, 4, measure_name)
    elif measure_name == "AbsMean":
        return compute_abs_mean(data)
    elif custom_measures_func is None:
        print("Unknown post measure or no module named custom_measures.py with a "
              "function compute_measures found. The measure", measure_name, "is not computed. "
              "You can compute custom measures by adding a corresponding .py file to your python path. In C++ "
              "this can be achieved by defining a python_modules_path via -DPYTHON_SCRIPTS_PATH=<relative or "
              "absolute path to custom_modules.py>.")
        return None, data
    else:
        try:
            # from custom_measures import compute_measures
            return custom_measures_func(data=data, measure_name=measure_name, custom_measures_args=custom_measures_args)
        except ModuleNotFoundError:
            print("Unknown post measure or no module named custom_measures.py with a "
                  "function compute_measures found. The measure", measure_name, "is not computed. "
                  "You can compute custom measures by adding a corresponding .py file to your python path. In C++ "
                  "this can be achieved by defining a python_modules_path via -DPYTHON_SCRIPTS_PATH=<relative or "
                  "absolute path to custom_modules.py>.")
            return None, data
        except Exception as e:
            print_exc()
            return None, data


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
    data.insert(len(data.columns), "AbsMean", np.abs(data["Config"].to_numpy().reshape(len(data), -1).mean(axis=1)))
    return ["AbsMean"], data


def compute_measures_over_config(data, measures, custom_measures_func=None, custom_measures_args=None):
    effective_measures = copy.deepcopy(measures)
    for measure in measures:
        if measure not in data.columns:
            new_measures, data = compute_measure_over_config(data=data, measure_name=measure, custom_measures_func=custom_measures_func, custom_measures_args=custom_measures_args)
            if new_measures is None: # No measure has been computed
                continue
            if len(new_measures) > 0:
                effective_measures.remove(measure)
                effective_measures += new_measures
    return effective_measures, data


""" One of rel_data_dir and data needs to be defined. """
def expectation_value(measures, running_parameter="default", rp_values=None, rel_data_dir=None, data=None,
                      number_of_measurements=None, error_type="statistical", n_means_bootstrap=None, rel_results_dir=None, sim_base_dir=None,
                      custom_measures_func=None, custom_measures_args=None, custom_load_data_func=None, custom_load_data_args=None):
    print("Computing expectation values...")

    # Load data
    if data is None:
        data = load_data_based_running_parameter(
            rel_data_dir=rel_data_dir, identifier="expectation_value", running_parameter=running_parameter, rp_values=rp_values,
            sim_base_dir=sim_base_dir, custom_load_data_func=custom_load_data_func, custom_load_data_args=custom_load_data_args)

    if number_of_measurements is not None:
        assert number_of_measurements <= len(data.loc[data.index.unique(0)[0]]), "Number of measurements cannot exceed number of actual measurements."
    else:
        number_of_measurements = len(data.loc[data.index.unique(0)[0]])

    # Compute measures based on the given configurations that have not been computed during the simulation
    post_measures = [measure for measure in measures if measure not in data.columns.unique(0).values]
    if len(post_measures) > 0:
        post_measures, data = compute_measures_over_config(
            data=data, measures=post_measures, custom_measures_func=custom_measures_func,
            custom_measures_args=custom_measures_args
        )

    # Compute the expectation values and the error
    ep = ExpectationValue(data=data)

    black_expectation_value_list = ["Config"]
    # expectation_value_measures = [
    #     exp_value for exp_value in np.union1d(measures, post_measures) if exp_value not in black_expectation_value_list and exp_value in ep.data.columns]
    expectation_value_measures = [
        exp_value for exp_value in np.union1d(measures, post_measures) if exp_value not in black_expectation_value_list]


    ep.compute_expectation_value(columns=expectation_value_measures,
                                 exp_values=['mean'])  # , 'max', 'min' , 'secondMoment', 'fourthMoment'
    expectation_values = ep.expectation_values

    if error_type == "bootstrap":
        ep.bootstrap_error(n_means_boostrap=n_means_bootstrap,
                           number_of_measurements=number_of_measurements,
                           columns=expectation_value_measures,
                           exp_values=['mean'],
                           running_parameter=running_parameter)
    elif error_type == "jackknife":
        ep.jackknife_error(number_of_measurements=number_of_measurements,
                           columns=expectation_value_measures,
                           exp_values=['mean'],
                           running_parameter=running_parameter)
    else:
        ep.statistical_error(number_of_measurements=number_of_measurements,
                             columns=expectation_value_measures)

    errors = ep.errors

    expectation_values = ep.drop_multiindex_levels_with_unique_entries(data=expectation_values)
    errors = ep.drop_multiindex_levels_with_unique_entries(data=errors)
    results = pd.concat([expectation_values, errors], keys=["Estimate", "Std"], names=["val"], axis=1)
    results = results.sort_index(axis=1, level=1, sort_remaining=False)

    if rel_results_dir is not None:
        # Combine results with existing results
        existing_json_results = load_expectation_value_results(rel_results_dir=rel_results_dir, sim_base_dir=sim_base_dir)
        if existing_json_results is not None:
            try:
                json_results = results.combine_first(existing_json_results)
            except ValueError:
                json_results = None
                print("Note: Results were not stored. Expectation values in existing expectation_value_results.json and computed expectation values "
                      "are based on different key input variables (measures, etc.). Either delete the existing json file "
                      "or adapt your simulation parameters.")
        else:
            json_results = results

        if json_results is not None:
            json_results = json_results.transpose()
            json_results = json_results.reset_index()
            json_results = json_results.transpose()
            json_results = json_results.reset_index()

            from mcmctools.utils.utils import get_rel_path
            results_path = get_rel_path(rel_dir=rel_results_dir, sim_base_dir=sim_base_dir)

            json_results = json_results.applymap(str)
            print("Generating expectation_value_results.json file in", os.path.abspath(results_path))
            json_results.to_json(results_path + "/expectation_value_results.json", indent=4)

    else:
        results = results.sort_index(axis=1, level=1, sort_remaining=False)

    return results


def load_expectation_value_results(rel_results_dir, sim_base_dir=None):
    from mcmctools.utils.utils import get_rel_path
    results_path = get_rel_path(rel_dir=rel_results_dir, sim_base_dir=sim_base_dir)

    if os.path.exists(results_path + "/expectation_value_results.json"):
        results = pd.read_json(
            results_path + "/expectation_value_results.json",
            convert_dates=False,  # dont convert columns to dates
            convert_axes=False  # dont convert index to dates
        )

        running_parameter = results.columns[0]

        results = results.set_index(running_parameter)

        column_levels = []
        for item_idx in results.index.values:
            if item_idx in ["val", "quantity", "elem", "component"]:
                column_levels.append(item_idx)
            else:
                break
        results = results.transpose()
        results = results.set_index(column_levels)
        results = results.transpose()
        # Convert strings in dataframe to numbers
        results = results.applymap(to_float)
        return results
    else:
        # No results found
        return None


if __name__ == '__main__':
    print("FilesDir:", sys.argv[1])  # , "SimRootDir:", sys.argv[2], "RelPath:", sys.argv[3])
    expectation_value(sys.argv[1])  # , sys.argv[2], sys.argv[3])
