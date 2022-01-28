import os
import sys
import numpy as np
import pandas as pd

from pystatplottools.visualization.utils import figure_decorator

from mcmctools.loading.loading import load_data_based_running_parameter


def get_correlation_time(data, tau, running_parameter):
    # For one dataset
    # n_samples = len(data) - tau
    # c_tau = data.iloc[tau:].reset_index(drop=True)
    # c = data.iloc[:n_samples].reset_index(drop=True)
    # return c.multiply(c_tau).mean() - c.mean() * c_tau.mean()

    c_tau = data.groupby(running_parameter).apply(lambda x: x.iloc[tau:].reset_index(drop=True))
    c = data.groupby(running_parameter).apply(lambda x: x.iloc[:len(x) - tau].reset_index(drop=True))
    return c.multiply(c_tau).groupby(running_parameter).mean() - c.groupby(running_parameter).mean() * c_tau.groupby(running_parameter).mean()


def determine_correlation_time(x, maximum_correlation_time):
    intersection = np.argwhere(x.values - np.exp(-1) < 0).flatten()
    if len(intersection) > 0:
        return intersection[0]
    else:
        return maximum_correlation_time - 1


@figure_decorator
def plot_correlation_time(corr_times, label, maximum_correlation_time, filename=None, fig=None, ax=None, fma=None, figsize=(10, 7), width=1.3, type="png", **kwargs):
    ax.plot(corr_times, label=label)
    ax.plot([0, maximum_correlation_time], [np.exp(-1), np.exp(-1)])
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$c(\\tau)$")
    ax.set_yscale('log')
    ax.legend(loc="upper right")


def correlation_time(minimum_sample_size, maximum_correlation_time, measure, running_parameter="default", rp_keys=None,
                     rel_data_dir=None, data=None, rel_results_dir=None, sim_base_dir=None, fma=None,
                     custom_load_data_func=None, custom_load_data_args=None):
    print("Computing correlation times...")

    # Load data
    if data is None:
        data = load_data_based_running_parameter(
            rel_data_dir=rel_data_dir, identifier="correlation_time", running_parameter=running_parameter,
            rp_keys=rp_keys, sim_base_dir=sim_base_dir, custom_load_data_func=custom_load_data_func,
            custom_load_data_args=custom_load_data_args)

    if running_parameter.capitalize() in data.columns:
        del data[running_parameter.capitalize()]

    from mcmctools.utils.utils import get_rel_path
    results_path = get_rel_path(rel_dir=rel_results_dir, sim_base_dir=sim_base_dir)

    correlation_times = np.array(
        [get_correlation_time(data, t, running_parameter).values for t in range(0, maximum_correlation_time)])

    correlation_times = pd.DataFrame(data=correlation_times.reshape(maximum_correlation_time, -1),
                                     columns=data.index.levels[0],
                                     index=pd.RangeIndex(start=0, stop=maximum_correlation_time, step=1))
    correlation_times = correlation_times.div(correlation_times.iloc[0])

    taus = correlation_times.apply(lambda x: determine_correlation_time(x, maximum_correlation_time=maximum_correlation_time), axis=0)
    taus = pd.DataFrame(data=taus + 1, columns=["CorrelationTime"])
    taus.index = taus.index.set_names(None)  # To be in concordance with loaded correlation times

    if running_parameter != "default":
        for label, corr_times in correlation_times.items():
            plot_correlation_time(
                corr_times, running_parameter + " = " + label, maximum_correlation_time, fma=fma,
                filename="correlation_time_" + running_parameter + "=" + label, directory=results_path)

    else:
        plot_correlation_time(correlation_times, "$c(\\tau)$", maximum_correlation_time, fma=fma,
                              filename="correlation_time", directory=results_path)

    if rel_results_dir is not None:
        # Combine taus with existing taus
        existing_taus = load_correlation_times_results(rel_results_dir=rel_results_dir,
                                                       sim_base_dir=sim_base_dir)
        if existing_taus is not None:
            try:
                json_taus = taus.combine_first(existing_taus)
            except ValueError:
                json_taus = taus
                print("Correlation time in existing correlation_time_results.json and computed correlation times "
                      "are based on different key input variables. Either delete the existing json file "
                      "or adapt necessary parameters.")
        else:
            json_taus = taus

        json_taus.to_json(results_path + "/correlation_time_results.json", indent=4)

    return taus


def load_correlation_times_results(rel_results_dir, sim_base_dir=None):
    from mcmctools.utils.utils import get_rel_path
    results_path = get_rel_path(rel_dir=rel_results_dir, sim_base_dir=sim_base_dir)

    if os.path.exists(results_path + "/correlation_time_results.json"):
        from pystatplottools.utils.utils import load_json
        correlation_times = pd.DataFrame(load_json(results_path + "/correlation_time_results.json"))
        return pd.DataFrame(correlation_times)
    else:
        # No results found
        print("Generating correlation_time_results.json file in", os.path.abspath(results_path))
        return None


if __name__ == '__main__':
    print("FilesDir:", sys.argv[1])  # , "SimRootDir:", sys.argv[2], "RelPath:", sys.argv[3])
    correlation_time(sys.argv[1])  # , sys.argv[2], sys.argv[3])
