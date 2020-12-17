from pystatplottools.ppd_pdf_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=False)

import os
import sys

from mcmctools.utils.json import load_configs
from mcmctools.loading.loading import load_data


def equilibriate(files_dir, sim_root_dir="", rel_path="./"):
    # Load configs and data
    cwd = os.getcwd()

    sim_params, execution_params, running_parameter = load_configs(files_dir=files_dir, mode="correlation_time")
    data, filenames = load_data(files_dir=files_dir, running_parameter=running_parameter, identifier="correlation_time")

    if running_parameter.capitalize() in data.columns:
        del data[running_parameter.capitalize()]

    os.chdir(cwd)
    assert False, "Equilibrium computation is in construction."


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("FilesDir:", sys.argv[1])  # , "SimRootDir:", sys.argv[2], "RelPath:", sys.argv[3])
        equilibriate(sys.argv[1])  # , sys.argv[2], sys.argv[3])
    else:
        pass