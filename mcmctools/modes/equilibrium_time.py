import os
import sys
import numpy as np
import pandas as pd

from pystatplottools.visualization.utils import figure_decorator

from mcmctools.loading.loading import load_data_based_running_parameter


@figure_decorator
def plot_equilibrium_time(mean_observables, equilibrium_time, label, filename=None, fig=None, ax=None, fma=None, figsize=(10, 7), width=1.3, type="png", **kwargs):
    ax.plot(np.arange(0, mean_observables.shape[0]), mean_observables[:, 0], label="hot")
    ax.plot(np.arange(0, mean_observables.shape[0]), mean_observables[:, 1], label="cold")

    lb, ub = ax.get_ylim()
    ax.plot([equilibrium_time, equilibrium_time], [lb, ub])
    ax.set_xlabel("$t$")
    ax.set_ylabel(label)
    ax.legend(loc="upper right")


def equilibrium_time(sample_size, number_of_steps, measure, confidence_range=0.1,
                     confidence_window=100, running_parameter="default", rp_values=None,
                     rel_data_dir=None, data=None, rel_results_dir=None,
                     sim_base_dir=None, fma=None, custom_load_data_func=None, custom_load_data_args=None):
    print("Computing equilibrium times...")

    # Load data
    if data is None:
        data = load_data_based_running_parameter(
            rel_data_dir=rel_data_dir, identifier="equilibrium_time", running_parameter=running_parameter,
            rp_values=rp_values, sim_base_dir=sim_base_dir,
            custom_load_data_func=custom_load_data_func, custom_load_data_args=custom_load_data_args)

    if running_parameter.capitalize() in data.columns:
        del data[running_parameter.capitalize()]

    from mcmctools.utils.utils import get_rel_path
    results_path = get_rel_path(rel_dir=rel_results_dir, sim_base_dir=sim_base_dir)

    n_cmv = max(number_of_steps, confidence_window) - min(number_of_steps, confidence_window) + 1

    running_parameters = data.index.unique(level=0).values
    equilibrium_times = np.zeros(len(running_parameters))
    for idx, running_parameter_value in enumerate(running_parameters):
        mean_observables = data.loc[running_parameter_value].values.reshape(sample_size, 2, number_of_steps).mean(axis=0).transpose()
        mean_observables[:n_cmv, 0] = np.convolve(mean_observables[:, 0], np.ones(confidence_window) * 1.0 / confidence_window, 'valid')
        mean_observables[:n_cmv, 1] = np.convolve(mean_observables[:, 1], np.ones(confidence_window) * 1.0 / confidence_window, 'valid')
        equilibrium_time = np.argwhere(np.abs(mean_observables[:n_cmv, 0] - mean_observables[:n_cmv, 1]) - confidence_range < 0).flatten()

        if len(equilibrium_time) == 0: # Never happend -> setting to maximum
            equilibrium_times[idx] = number_of_steps
        else:
            equilibrium_times[idx] = equilibrium_time[0]

        if running_parameter != "default":
            plot_equilibrium_time(
                mean_observables=mean_observables[:n_cmv], equilibrium_time=equilibrium_times[idx], label=measure,
                fma=fma, filename="equilibrium_time_" + running_parameter + "=" + running_parameter_value,
                directory=results_path)
        else:
            plot_equilibrium_time(
                mean_observables=mean_observables[:n_cmv], equilibrium_time=equilibrium_times[idx],
                label=measure, fma=fma, filename="equilibrium_time", directory=results_path)

    equilibrium_times = pd.DataFrame(data=equilibrium_times, index=running_parameters, columns=["EquilibriumTime"])

    if rel_results_dir is not None:
        # Combine results with existing results
        existing_equilibrium_times = load_equilibrium_times_results(rel_results_dir=rel_results_dir,
                                                                    sim_base_dir=sim_base_dir)
        if existing_equilibrium_times is not None:
            try:
                json_equilibrium_times = equilibrium_times.combine_first(existing_equilibrium_times)
            except ValueError:
                json_equilibrium_times = equilibrium_times
                print("Equilibrium time in existing equilibrium_time_results.json and computed equilibrium times "
                      "are based on different key input variables. Either delete the existing json file "
                      "or adapt necessary parameters.")
        else:
            json_equilibrium_times = equilibrium_times

        json_equilibrium_times.to_json(results_path + "/equilibrium_time_results.json", indent=4)

    return equilibrium_times


def load_equilibrium_times_results(rel_results_dir, sim_base_dir=None):
    from mcmctools.utils.utils import get_rel_path
    results_path = get_rel_path(rel_dir=rel_results_dir, sim_base_dir=sim_base_dir)

    if os.path.exists(results_path + "/equilibrium_time_results.json"):
        from pystatplottools.utils.utils import load_json
        equilibrium_times = pd.DataFrame(load_json(results_path + "/equilibrium_time_results.json"))
        return pd.DataFrame(equilibrium_times)
    else:
        # No results found
        print("Generating equilibrium_time_results.json file in",  os.path.abspath(results_path))
        return None


if __name__ == '__main__':
    # print("FilesDir:", sys.argv[1])  # , "SimRootDir:", sys.argv[2], "RelPath:", sys.argv[3])
    # equilibrium_time(sys.argv[1])  # , sys.argv[2], sys.argv[3])
    os.chdir('./../../../LatticeModelSimulationLib/examples/examples/IsingModel/')
    equilibrium_time("IsingModelMetropolis")
