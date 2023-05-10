import tensorflow as tf
import numpy as np
from train import train_pinn, train_xtfc
from create_figures import create_figures
import itertools

import pdb


# parameters for the physical tidal system
params= [
    2000, # xm
    48, # tm
    0, #hz
    0.001, # S
    2000/24, # T
    0.65, # A
    (2 *np.pi) / 24, # angular velocity
    0, # phase shift
    0, # leakance
]

# parameters used in create_figures.py
eval_params = [
    24,
    96,
]

#TODO get rid of all references to args

# specify values for each hyperparameter
# TODO specify good values for each hyperparameter
## choose which hyperparameters I want to sweep for each method
### hidden layer size
### number of epochs
### n_train

#TODO add random see as a hyperparameter
tf.random.set_seed(12321)

# sweep a single hyperparameter (test case)
# experiment_params = {
#     "model": ["pinn"],
#     "layers": [[16, 32, 32, 16]],
#     "n_train": [1000],
#     "n_epochs": [5],
#     "lr": [1e-3, 1e-4],
# }

# the only parameters used by xtfc is the size of the hidden layer
## otherwise, it trains using iterative least squares, so no hyperparameters there
# experiment_params = {
#     "model": ["xtfc"],
#     "layers": [16],
#     "n_train": [1000],
# }

# sweep multiple hyperparameters
# experiment_params = {
#     "model": ["pinn"],
#     "layers": [[16, 32, 32, 16]],
#     "n_train": [50000, 100000],
#     "n_epochs": [50],
#     "lr": [1e-2, 1e-3],
# }

# experiment_params = {
#     "model": ["pinn"],
#     "layers": [[16, 32, 32, 16]],
#     "n_train": [50000, 100000, 200000],
#     "n_epochs": [30],
#     "lr": [1e-3],
# }

# experiment_params = {
#     "model": ["pinn"],
#     "layers": [[64, 128, 128, 64]],
#     "n_train": [50000],
#     "n_epochs": [30],
#     "lr": [1e-3],
# }

xtfc_params= [1,1,0.001,2000/24,0.65,2*np.pi/24,24]

experiment_params = {
    "model": ["xtfc"],
    "layers": [50, 75, 100],
}



# get all possible combinations of hyperparameters
keys, values = zip(*experiment_params.items())
hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# get unique strings for each hyperparam dict
hyperparam_dicts = {}
for i in hyperparam_combinations:
    hyperparam_str = ""
    for k, v in i.items():
        hyperparam_str += f"{str(k)}_{str(v)}_"
    hyperparam_str = hyperparam_str[:-1]
    print(hyperparam_str)
    hyperparam_dicts[hyperparam_str] = i

# training loop where each combination of hyperparameters is used to train a different network
networks = {}
for hyperparam_str, hyperparam_dict in hyperparam_dicts.items():
    # case of PINN
    if hyperparam_dict["model"] == "pinn":
        network = train_pinn(params, hyperparam_dict)
        localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
        network.save(f"pinn_tidal/models/model_{hyperparam_str}.keras", options=localhost_save_option)
        # tf.keras.saving.save_model(model=network, filepath=f"./pinn_tidal/models/model_{hyperparam_str}.ckpt")
    # case of X-TFC
    elif hyperparam_dict["model"] == "xtfc":
        train_xtfc(xtfc_params, hyperparam_dict)
        #TODO do stuff to train and evaluate these models
        pass


# create_figures(network, params, eval_params, n_test=2000)
