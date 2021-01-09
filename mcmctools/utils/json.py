import os

from pystatplottools.utils.utils import load_json


def load_json_params(path, param_name, project_base_dir):
    params = load_json(path + "/" + param_name + ".json")
    for key in list(params.keys()):
        if "params_path" in key and key != "execution_params_path":
            params[key[:-5]] = load_json_params(project_base_dir + "/" + params[key] + "/", key[:-5], project_base_dir=project_base_dir)
    return params


def load_configs(files_dir, mode, project_base_dir):
    config_path = project_base_dir + "/configs/" + files_dir

    sim_params = load_json_params(config_path, "sim_params", project_base_dir=project_base_dir)
    execution_params = load_json_params(config_path, mode + "_params", project_base_dir=project_base_dir)

    if sim_params["running_parameter"] == 'None':
        running_parameter = "default"
    else:
        running_parameter = sim_params["running_parameter"]
    return sim_params, execution_params, running_parameter