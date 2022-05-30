import yaml 
import csv 
import os 
from argparse import Namespace
import shutil
import numpy as np
from numba import njit
from matplotlib import pyplot as plt


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

def init_reward_struct(path):
    if os.path.exists(path):
        return 
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

    id = 0
    run_list = []
    set_n = run_dict['set_n']
    for rep in range(run_dict['n']):
        for run in run_dict['runs']:
            run["n"] = rep
            run["id"] = id
            run["set_n"] = set_n
            run['run_name'] = f"{run_dict['test_name']}_{run['map_name']}_{set_n}_{rep}_{id}"
            run['reward_name'] = run_dict['reward_name']
            run['path'] = f"{run_dict['test_name']}/"
            run['test_name'] = f"{run_dict['test_name']}"
            run["discrete"] = False #! dangerous

            run_list.append(Namespace(**run))
            id += 1

    test_params = {}
    for key in run_dict.keys():
        if key != "runs":
            test_params[key] = run_dict[key]
    test_params = Namespace(**test_params)

    init_reward_struct("Data/Vehicles/" + run_list[0].path)

    return run_list, test_params



@njit(cache=True)
def calculate_speed(delta, f_s=0.9):
    b = 0.523
    g = 9.81
    l_d = 0.329
    # f_s = 0.7
    max_v = 6

    if abs(delta) < 0.03:
        return max_v

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    V = min(V, max_v)

    return V

def plot_speed_profile():
    ds = np.linspace(-0.4, 0.4, 50)

    plt.figure(1)
    for fs in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        vs = np.array([calculate_speed(d, fs) for d in ds])
        plt.plot(ds, vs)
    plt.show()

if __name__ == '__main__':


    plot_speed_profile()
