from pystatplottools.pdf_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=False)

import os
import sys

from mcmctools.utils.json import load_configs
from mcmctools.loading.loading import load_data
from pystatplottools.distributions.distributionDD import DistributionDD


def site_distribution(files_dir, data, config_params):
    axes_indices = [config_params["xkey"], config_params["ykey"]]

    dist2d = DistributionDD(data=data)

    # Generate for each given column a two d distribution based on x_index and y_index as columns
    # This is done separately for each initial dataset
    binned_statistics = dist2d.binned_statistics_over_axes(
        axes_indices=axes_indices,
        range_min=[config_params["rmin_x"], config_params["rmin_y"]],
        range_max=[config_params["rmax_x"], config_params["rmax_y"]],
        nbins=[100, 100],
        statistic='probability'
    )

    z_index = "probability"

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=axes_indices,
                                                                       binned_statistics=binned_statistics,
                                                                       output_column_names=z_index)

    dataframe_indices = linearized_statistics.index.unique(0)
    print("Considered dataframes", dataframe_indices.values)

    from pystatplottools.plotting_env.contour2D import Contour2D
    contour2D = Contour2D(
        data=linearized_statistics.loc["default"],
        compute_x_func=lambda x: x[axes_indices[0]],  # possibility to rescale x and y axis or perform other operation for x axis
        # like computing a mass difference
        compute_y_func=lambda x: x[axes_indices[1]],
        z_index=z_index
    )

    fig, ax = fma.newfig(1.4)
    contour2D.set_ax_labels(ax, x_label="x", y_label="y")
    cf = contour2D.contourf(
        ax=ax,
        cbar_scale="Lin",
        lev_num=20,
    )
    contour2D.add_colorbar(fig=fig, cf=cf, z_label="Probability"
                           # cax=cbar_ax,
                           # z_ticks=[-1.4, -1, -0.5, 0, 0.5, 1, 1.4],
                           # z_tick_labels=['$-1.4$', '$-1$', '$-0.5$', '$0$', '$0.5$',
                           #               '$1$', '$1.4$']
                           )
    plt.tight_layout()

    filename = "complex_distribution"
    fma.savefig(os.getcwd() + "/results/" + files_dir, filename)


def plot_site_distribution(files_dir, sim_root_dir="", rel_path="./"):
    assert False, "Plot site distribution is still under construction"

    # Load configs and data
    cwd = os.getcwd()

    sim_params, execution_params, running_parameter = load_configs(files_dir=files_dir, mode="plot_site_distribution", project_base_dir=cwd)
    data, filenames = load_data(files_dir=files_dir, running_parameter=running_parameter, identifier="expectation_value")

    site_distribution(files_dir=files_dir, data=data, config_params=execution_params)

    os.chdir(cwd)


def custom_site_distribution(files_dir, sigma):
    config_params = {"xkey": "StateReal", "ykey": "StateImag", "rmin_x": -5.0, "rmax_x": 5.0, "rmin_y": -5.0, "rmax_y": 5.0}
    data_path = os.getcwd() + "/data/" + files_dir + "_" + f"{sigma[0]:.6f}" + "_" + f"{sigma[1]:.6f}" + "/"
    from mcmctools.loading.loading import ConfigurationLoader
    data, filenames = ConfigurationLoader.load_all_configurations(
        path=data_path,
        identifier="expectation_value",
        running_parameter="default")
    site_distribution(files_dir=files_dir, data=data, config_params=config_params)

#
# def load_plot_site_distribution_results(filename, directory, root_dir, rel_path="./"):
#     from wand.image import Image as WImage
#     img = WImage(
#         filename=rel_path + root_dir + "./../plots/" + directory + "/expectation_value_" + filename + "_contour.pdf",
#         resolution=160)
#     return img


if __name__ == '__main__':
    print("FilesDir:", sys.argv[1])  # , "SimRootDir:", sys.argv[2], "RelPath:", sys.argv[3])
    plot_site_distribution(sys.argv[1])  # , sys.argv[2], sys.argv[3])
