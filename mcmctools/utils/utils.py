import numpy as np


def linspace_rp_intervals(rp_min, rp_max, rp_num):
    return np.array([np.round(rp_min + i * (rp_max - rp_min) / (rp_num - 1), 8) for i in range(rp_num)])


def to_float(x):
    if "j" in x:
        return np.complex(x)
    else:
        try:
            return float(x)
        except:
            return x


def get_rel_path(rel_dir=None, sim_base_dir=None):
    import os
    if rel_dir is not None:
        if sim_base_dir is None:
            rel_path = os.getcwd() + "/" + rel_dir
        else:
            rel_path = sim_base_dir + "/" + rel_dir

        if not os.path.isdir(rel_path):
            os.makedirs(rel_path)
    else:
        rel_path = None
    return rel_path
