import os

from pystatplottools.utils.utils import load_json


def load_json_params(path, param_name, sim_base_dir):
    params = load_json(path + "/" + param_name + ".json")
    for key in list(params.keys()):
        if "params_path" in key and key != "execution_params_path":
            params[key[:-5]] = load_json_params(sim_base_dir + "/" + params[key] + "/", key[:-5], sim_base_dir=sim_base_dir)
    return params


def load_configs(files_dir, mode, sim_base_dir):
    config_path = sim_base_dir + "/configs/" + files_dir

    sim_params = load_json_params(config_path, "sim_params", sim_base_dir=sim_base_dir)
    execution_params = load_json_params(config_path, mode + "_params", sim_base_dir=sim_base_dir)

    if sim_params["running_parameter"] == 'None':
        running_parameter = "default"
    else:
        running_parameter = sim_params["running_parameter"]
    return sim_params, execution_params, running_parameter


def load_json_sim_params(rel_data_path, identifier, running_parameter, rp_key, sim_base_dir):
    import os
    if sim_base_dir is None:
        data_path = os.getcwd() + "/" + rel_data_path
    else:
        data_path = sim_base_dir + "/" + rel_data_path
    return load_json(data_path + "/" + identifier + "_" + running_parameter + "=" + f"{rp_key:.6f}" + ".json")
