import os


from pystatplottools.ppd_utils.utils import load_json


def load_json_params(path, param_name):
    params = load_json(path + "/" + param_name + ".json")
    for key in list(params.keys()):
        if "params_path" in key and key != "execution_params_path":
            params[key[:-5]] = load_json_params(os.getcwd() + "/" + params[key] + "/", key[:-5])
    return params


def load_configs(files_dir, mode):
    config_path = os.getcwd() + "/configs/" + files_dir

    sim_params = load_json_params(config_path, "sim_params")
    execution_params = load_json_params(config_path, mode + "_params")

    if sim_params["running_parameter"] == 'None':
        running_parameter = "default"
    else:
        running_parameter = sim_params["running_parameter"]
    return sim_params, execution_params, running_parameter