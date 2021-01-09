from pystatplottools.pdf_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=False)

import os
import sys
import numpy as np
import pandas as pd

from mcmctools.utils.json import load_configs
from mcmctools.loading.loading import load_data


def get_correlation_time(data, tau, running_parameter):
    # For one dataset
    # n_samples = len(data) - tau
    # c_tau = data.iloc[tau:].reset_index(drop=True)
    # c = data.iloc[:n_samples].reset_index(drop=True)
    # return c.multiply(c_tau).mean() - c.mean() * c_tau.mean()

    c_tau = data.groupby(running_parameter).apply(lambda x: x.iloc[tau:].reset_index(drop=True))
    c = data.groupby(running_parameter).apply(lambda x: x.iloc[:len(x) - tau].reset_index(drop=True))
    return c.multiply(c_tau).groupby(running_parameter).mean() - c.groupby(running_parameter).mean() * c_tau.groupby(running_parameter).mean()


def plot_correlation_time(corr_times, label, maximum_correlation_time, files_dir, filename):
    fig, ax = fma.newfig(1)

    ax.plot(corr_times, label=label)
    ax.plot([0, maximum_correlation_time], [np.exp(-1), np.exp(-1)])
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$c(\\tau)$")
    ax.set_yscale('log')
    ax.legend(loc="upper right")

    plt.tight_layout()

    fma.savefig(os.getcwd() + "/results/" + files_dir, filename)
    plt.close()


def correlation_time(files_dir, sim_root_dir="", rel_path="./"):
    # Load configs and data
    cwd = os.getcwd()

    sim_params, execution_params, running_parameter = load_configs(files_dir=files_dir, mode="correlation_time", project_base_dir=cwd)
    data, filenames = load_data(files_dir=files_dir, running_parameter=running_parameter, identifier="correlation_time")

    if running_parameter.capitalize() in data.columns:
        del data[running_parameter.capitalize()]

    correlation_times = np.array(
        [get_correlation_time(data, t, running_parameter).values for t in range(0, execution_params["maximum_correlation_time"])])

    correlation_times = pd.DataFrame(data=correlation_times.reshape(execution_params["maximum_correlation_time"], -1),
                                     columns=data.index.levels[0],
                                     index=pd.RangeIndex(start=0, stop=execution_params["maximum_correlation_time"], step=1))
    correlation_times = correlation_times.div(correlation_times.iloc[0])

    taus = correlation_times.apply(lambda x: np.argmin(np.abs(x.values - np.exp(-1)), axis=0))
    taus = pd.DataFrame(data=taus, columns=["CorrelationTime"])

    if running_parameter != "default":
        for label, corr_times in correlation_times.items():
            plot_correlation_time(corr_times, running_parameter + " = " + label, execution_params["maximum_correlation_time"], files_dir,
                                  "correlation_time_" + running_parameter + "=" + label)

    else:
        plot_correlation_time(correlation_times, "$c(\\tau)$", execution_params["maximum_correlation_time"], files_dir,
                              "correlation_time")

    if not os.path.isdir(os.getcwd() + "/results/" + files_dir):
        os.makedirs(os.getcwd() + "/results/" + files_dir)

    taus.to_json(os.getcwd() + "/results/" + files_dir + "/correlation_time_results.json", indent=4)

    os.chdir(cwd)


if __name__ == '__main__':
    print("FilesDir:", sys.argv[1])  # , "SimRootDir:", sys.argv[2], "RelPath:", sys.argv[3])
    correlation_time(sys.argv[1])  # , sys.argv[2], sys.argv[3])
