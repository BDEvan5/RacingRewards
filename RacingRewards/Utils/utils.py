import yaml 
import csv 
import os 
from argparse import Namespace
import shutil
import numpy as np
from numba import njit

# Admin functions
def save_conf_dict(dictionary, save_name=None):
    if save_name is None:
        save_name  = dictionary["run_name"]
    path = "Data/Vehicles/" + dictionary["path"] + dictionary["run_name"] + f"/{save_name}_record.yaml"
    with open(path, 'w') as file:
        yaml.dump(dictionary, file)

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    np.random.seed(conf.random_seed)

    return conf

def load_run_list(fname):
    full_path =  "run_files/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    return conf_dict



def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)

@njit(cache=True)
def limit_phi(phi):
    while phi > np.pi:
        phi = phi - 2*np.pi
    while phi < -np.pi:
        phi = phi + 2*np.pi
    return phi


# def setup_run_list(run_file):
#     run_dict = load_run_list(run_file)

#     run_list = []
#     run_n = 0
#     for rep in range(run_dict['n']):
#         for run in run_dict['runs']:
#             run["id"] = run_n
#             run["n"] = rep
#             run['run_name'] = f"{run_dict['reward_name']}_{run['map_name']}_{run['r_ct']}_{run['r_head']}_{rep}_{run_n}"
#             run['path'] = f"{run_dict['reward_name']}/"

#             run_list.append(Namespace(**run))
#             run_n += 1

#     return run_list



def setup_run_list(run_file):
    run_dict = load_run_list(run_file)

    run_list = []
    run_n = 0
    for rep in range(run_dict['n']):
        for run in run_dict['runs']:
            run["id"] = run_n
            run["n"] = rep
            run['run_name'] = f"{run_dict['reward_name']}_{run['map_name']}_{rep}_{run_n}"
            run['path'] = f"{run_dict['reward_name']}/"

            run_list.append(Namespace(**run))
            run_n += 1

    return run_list


