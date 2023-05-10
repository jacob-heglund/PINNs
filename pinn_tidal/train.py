import tensorflow as tf
import numpy as np
from model import Network, PINN, L_BFGS_B, Adam
from analytical_solution import h0

import sympy as sp
from sympy import Matrix, diff, symbols, tanh,cos,exp,tan,pi, lambdify
from sympy.abc import t, x
from scipy.sparse.linalg import lsqr

import time

import pdb

"""
physical parameters for the tidal system

params[0] = xm, max distance from the land
params[1] = tm, max time for the simulation
params[2] = hz, initial head at t=0
params[3] = S, storativity, a dimensionless measure of the volume of water that will be discharged from an aquifer per unit area of the aquifer and per unit reduction in hydraulic head
params[4] = T, transmissivity
params[5] = A, amplitude of tidal changes
params[6] = a, angular velocity
params[7] = c, phase shift of tidal changes
params[8] = L, leakance / specific leakage
"""

def train_pinn(params, hyperparams):
    """
    Train the physics informed neural network (PINN) model for the wave equation.
    """

    # build a core network model
    network = Network.build(layers=hyperparams["layers"])
    network.summary()

    # build a PINN model
    pinn = PINN(network, params).build()

    # NOTE commented these out b/c these aren't used later in the code and take forever to run
    # sampler = qmc.LatinHypercube(d=2)
    # sample = sampler.random(n = hyperparams["n_train"])
    # qmc.discrepancy(sample)

    # create training input
    tx_eqn = np.zeros((hyperparams["n_train"], 2))
    tx_eqn[..., 0] = np.random.uniform(0, params[1], hyperparams["n_train"])
    tx_eqn[..., 1] = np.random.uniform(0, params[0], hyperparams["n_train"])

    tx_ini = np.zeros((hyperparams["n_train"], 2))
    tx_ini[..., 1] = np.random.uniform(0, params[0], hyperparams["n_train"])

    # uniformly sample t values for x = 0 boundary condition
    tx_bnd1 = np.zeros((hyperparams["n_train"], 2))
    tx_bnd1[..., 0] = np.random.uniform(0, params[1], hyperparams["n_train"])

    # uniformly sample t values for x = xm boundary condition
    tx_bnd2 = np.ones((hyperparams["n_train"], 2))
    tx_bnd2[..., 0] = np.random.uniform(0, params[1], hyperparams["n_train"])
    tx_bnd2[..., 1] = params[0] * tx_bnd2[..., 1]

    # create training output
    h_zero = np.zeros((hyperparams["n_train"], 1))
    h_bnd1 = h0(tf.constant(tx_bnd1), params).numpy()

    # create training data
    x_train = [tx_eqn, tx_bnd1, tx_bnd2]
    y_train = [h_zero, h_bnd1, h_zero]

    # train model with Adam
    Adam_optimizer = Adam(pinn)
    Adam_optimizer.compile(optimizer = \
                        tf.keras.optimizers.Adam(
                            learning_rate=hyperparams["lr"],
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-08,
                            name='Adam'),
                        loss = tf.keras.losses.MeanSquaredError(),
                        metrics=["mae"])

    Adam_optimizer.fit(x_train, y_train, epochs=hyperparams["n_epochs"])

    # # train model with L-BFGS-B
    ## NOTE this takes much longer than the SGD optimization step. We don't need this step to answer questions related to uncertainty, so leave it out.
    # lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    # lbfgs.fit()

    return network


# define fixed-weight matrix
def fixed_weight_matrix(fan_in,fan_out):
    w = np.random.uniform(low=- 1, high = 1, size=(fan_in,fan_out))
    w_sp = Matrix(w)
    return w_sp


# define input vector
def single_HLM_TD(fan_in,fan_out,t,x,params):
    w = fixed_weight_matrix(fan_in,fan_out)
    t,x = symbols("t,x")
    x_ = Matrix([[t],[x]])
    sigma0 = w.T*x_
    #sigma =sigma0.applyfunc(tanh)
    sigma = sigma0.applyfunc(tanh)
    sigma_xx = diff(sigma,x,2)
    sigma_t = diff(sigma,t)
    sigma_xx_0x = sigma_xx.subs(t,0)
    sigma_t_t0 = sigma_t.subs(x,0)
    sigma_t_tL = sigma_t.subs(x,params[0])
    sigma1 = sigma_xx-sigma_xx_0x
    sigma2 = sigma_t+(x-1)*sigma_t_t0-x*sigma_t_tL

    SHLM = params[2]/params[6]*sigma2.T-4*params[3]/pi**2*(cos(pi*x/2))**4*sigma1.T
    p = (params[5]*params[2]/2/params[3])**0.5
    a = params[4]*exp(-p*tan(pi*x/2))*cos(-p*tan(pi*x/2))+params[4]*(1-x)*(cos(params[5]*params[6]*t)-1)
    a_xx = diff(a,x,2)
    a_t = diff(a,t)
    b = 4*params[3]/pi**2*(cos(pi*x/2))**4*a_xx-params[2]/params[6]*a_t
    return SHLM,b,sigma,a


def train_xtfc(params, hyperparams):
    start = time.time()

    # compute single data hidden layer matrix and bias
    num_train_samples = 100
    tx_eqn = np.zeros((num_train_samples,2))
    tx_eqn[...,0]=np.linspace(0, params[1], num_train_samples)
    tx_eqn[...,1]=np.linspace(0, 1, num_train_samples)
    HLM = []
    b_1 = []
    fan_out = 100
    M,b,sigma,a = single_HLM_TD(2,fan_out,t,x,params)

    end = time.time()
    print(f"here 0, t = {end - start}")
    start = time.time()

    # compute num_samples of samples hidden layer matrix and bias
    for i in range(len(tx_eqn)):
        ti,xi = tx_eqn[...,0][i],tx_eqn[...,1][i]
        Mi = M.subs(t,ti).subs(x,xi)
        bi = b.subs(t,ti).subs(x,xi)
        c = (t,x)
        M_fi = lambdify(c, Mi, modules='numpy')
        b_fi = lambdify(c, bi, modules='numpy')
        M_i = M_fi(ti,xi).reshape(1,-1)
        b_i = b_fi(ti,xi)
        HLM.append(M_i)
        b_1.append(b_i)
    HLM_ = np.array(HLM).reshape(num_train_samples,fan_out)
    b_ = np.array(b_1).reshape(-1,1)
    b_[99] = 0

    end = time.time()
    print(f"here 1, t = {end - start}")
    start = time.time()

    m = np.linalg.lstsq(HLM_, b_, rcond=None)[0]
    cp = np.dot(HLM_,m)-b_
    beta_m = Matrix(m)

    return sigma, beta_m




"""
variables I need to save

sigma
beta_m

"""