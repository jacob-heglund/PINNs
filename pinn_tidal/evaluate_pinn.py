import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.gridspec import GridSpec
import os
import pandas as pd
import ast

import pdb


def evaluate_pinn(n_test=100000):
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

    # parameters used to create figures
    eval_params = [
        24,
        48,
    ]

    # load networks to evaluate
    networks = {}
    for fn in os.listdir("pinn_tidal/models/eval"):
        network_curr = tf.keras.models.load_model(f"pinn_tidal/models/eval/{fn}")
        network_str = fn.replace(".keras", "")
        networks[network_str] = network_curr

    cols = ["layer_size", "n_train", "x_val", "error"]
    output = pd.DataFrame(columns=cols)

    # make x cross-sections plots
    x_cross_sections = [100, 200, 300]

    for x_cs in x_cross_sections:
        # generate a vector of times
        n_test_t = n_test
        t_flat0 = np.linspace(0, (eval_params[0] / 2) - 1, int(n_test_t / (eval_params[1] / eval_params[0] * 2)))
        t_flat1 = np.tile(t_flat0, int(eval_params[1] / eval_params[0] * 2)).flatten()
        t_flat =  np.zeros_like(t_flat1)

        for i in range(len(t_flat)):
            if i < int(n_test_t / (eval_params[1] / eval_params[0] * 2)):
                t_flat[i] = t_flat1[i]
            else:
                t_flat[i] = round(t_flat1[i] + 12 * (i // (len(t_flat0))), 5)

        x_vals = x_cs * np.ones(n_test)

        # get analytical solution for h(t, x=const)
        p = 1 / (2 ** 0.5) * (((params[8] / params[4]) ** 2 + (params[6] * params[3] / params[4]) ** 2) ** 0.5 + params[8] / params[4]) ** 0.5
        h_analytical =  np.ones_like(x_vals) * params[2] + params[5] * np.exp(-p * x_vals) * np.cos(params[6] * t_flat - params[6] * params[3] / (2 * p * params[4]) * x_vals + params[7])

        #TODO actually get results over multiple networks and compare to get epistemic uncertainty

        for k, _ in networks.items():
            network = networks[k]

            # get network prediction for h(t, x=const)
            tx1 = np.stack([t_flat, x_vals/2], axis=-1)
            h_network = network.predict(tx1, batch_size=n_test)
            h_network = h_network.squeeze()

            # get error metric compared to analytical solution for this network
            ## this is not the same loss function used during training, so we expect loss_analytical to be different from the final training loss
            error = tf.keras.metrics.mean_absolute_error(h_network, h_analytical).numpy()
            # ["layer_size", "n_train", "x_val", "error"]
            start, end = k.find("["), k.find("]")
            layer_size_str = k[start:end+1]
            layer_size = ast.literal_eval(layer_size_str)[0]

            start = k.find("n_train") + 8
            tmp = k[start:]
            end = tmp.find("_") + start
            n_train_str = k[start:end]
            n_train = ast.literal_eval(n_train_str)

            output.loc[len(output.index)] = [layer_size, n_train, x_cs, error]

        """
        # plot h(t=const, x) vs x at various values of t
        plt.plot(t_flat, h_analytical, label="Analytical")
        plt.plot(t_flat, h_network, label="PINN")
        plt.legend()
        plt.xlabel("t")
        plt.savefig("fig.png")
        pdb.set_trace()
        """

    #RQ: Given a PINN model with all else held constant, how does the epistemic uncertainty change with different layer size?
    df = output.copy()

    df_layer_size = pd.DataFrame(columns=["layer_size", "variance"])
    for i in np.unique(df.layer_size):
        df_filter = df[df.layer_size ==  i]
        var = np.var(df_filter.error)
        df_layer_size.loc[len(df_layer_size.index)] = [i, var]

    #RQ: Given a PINN model with all else held constant, how does the epistemic uncertainty change with different amount of training data?
    df_n_train = pd.DataFrame(columns=["n_train", "variance"])
    for i in np.unique(df.n_train):
        df_filter = df[df.n_train ==  i]
        var = np.var(df_filter.error)
        df_n_train.loc[len(df_n_train.index)] = [i, var]

    df_total_var = pd.DataFrame(columns=["x_val", "variance"])
    for i in np.unique(df.x_val):
        df_filter = df[df.x_val ==  i]
        var = np.var(df_filter.error)
        df_total_var.loc[len(df_total_var.index)] = [i, var]

    pdb.set_trace()


    """
    #TODO actually make the right plot here from
    t_cross_sections = [0]
    for t_cs in t_cross_sections:
        # get analytical solution for h(t, x)
        p = 1 / (2 ** 0.5) * (((params[8] / params[4]) ** 2 + (params[6] * params[3] / params[4]) ** 2) ** 0.5 + params[8] / params[4]) ** 0.5
        x1 = np.linspace(0, params[0], n_test)
        h_analytical =  np.ones_like(x1) * params[2] + params[5] * np.exp(-p * x1) * np.cos(params[6] * t_cs - params[6] * params[3] / (2 * p * params[4]) * x1 + params[7])

        for k, _ in networks.items():
            network = networks[k]

        # get network prediction for h(t,x)
        n_test_t = n_test
        t_flat0 = np.linspace(0, (eval_params[0] / 2) - 1, int(n_test_t / (eval_params[1] / eval_params[0] * 2)))
        ho  = int(eval_params[1] / eval_params[0] * 2)
        t_flat1 = np.tile(t_flat0, int(eval_params[1] / eval_params[0] * 2)).flatten()
        t_flat =  np.zeros_like(t_flat1)

        for i in range(len(t_flat)):
            if i < int(n_test_t / (eval_params[1] / eval_params[0] * 2)):
                t_flat[i] = t_flat1[i]
            else:
                t_flat[i] = round(t_flat1[i] + 12 * (i // (len(t_flat0))), 5)

        x_flat = np.linspace(0, params[0], n_test)

        if 0 <= t_cs < eval_params[0] / 2:
            tx1 = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
            h_network = network.predict(tx1, batch_size=n_test)
        else:
            tx1 = np.stack([np.full(t_flat.shape, t_cs %  (eval_params[0] / 2)), x_flat], axis=-1)
            h_network = (-1)**(t_cs // (eval_params[0] / 2)) * network.predict(tx1, batch_size=n_test)
        pdb.set_trace()
        h_network = h_network.squeeze()

        # get error metric compared to analytical solution for this network
        ## this is not the same loss function used during training, so we expect loss_analytical to be different from the final training loss
        error = tf.keras.metrics.mean_absolute_error(h_network, h_analytical).numpy()

        #TODO actually get the right values from the network

        # plot h(t=const, x) vs x at various values of t
        plt.plot(t_flat, h_analytical, label="Analytical")
        plt.plot(t_flat, h_network, label="PINN")
        plt.legend()
        plt.xlabel("t")
        plt.savefig("fig.png")
        """



if __name__ == "__main__":
    evaluate_pinn()
