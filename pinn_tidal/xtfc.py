import sympy as sp
from sympy import Matrix
import numpy as np
from sympy import diff
from sympy import symbols
from sympy import tanh,cos,exp,tan,pi
from sympy import lambdify
from sympy.abc import t, x
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
#params=[xm,tm,hz,S,T,A,a,c,L]
#params=[0,  1, 2,3,4,5,6,7,8]

# new_params = [em,taom,S,T,A,a,tm]
params= [1,1,0.001,2000/24,0.65,2*np.pi/24,24]

# Input dimension is 2, output dimension :150-170, number of neurons in hidden layer



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


# compute single data hidden layer matrix and bias
num_train_samples = 100
tx_eqn = np.zeros((num_train_samples,2))
tx_eqn[...,0]=np.linspace(0, params[1], num_train_samples)
tx_eqn[...,1]=np.linspace(0, 1, num_train_samples)
HLM = []
b_1 = []
fan_out = 100
M,b,sigma,a = single_HLM_TD(2,fan_out,t,x,params)

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

m= np.linalg.lstsq(HLM_, b_, rcond=None)[0]
# print(m)
# solve for beta, least square
#beta = np.dot(np.linalg.inv(np.dot(HLM_.T,HLM_)),np.dot(HLM_.T,b_))
#beta = np.dot(np.linalg.inv(HLM_),b_)
cp = np.dot(HLM_,m)-b_
beta_m = Matrix(m)

# h as function of t and x
h_m = sigma.T*beta_m+(x-1)*sigma.T.subs(x,0)*beta_m-x*sigma.T.subs(x,1)*beta_m-x*sigma.T.subs(t,0).subs(x,0)*beta_m+x*sigma.T.subs(t,0).subs(x,params[0])*beta_m-sigma.T.subs(t,0)*beta_m+sigma.T.subs(t,0).subs(x,0)*beta_m
h_ele = h_m[0,0]
h = h_ele +a


num_test_samples = 100
tx_eqn1 = np.zeros((num_test_samples,2))
tx_eqn1[...,0]= 1/4
tx_eqn1[...,1]=np.linspace(0, 1, num_test_samples)

h_solution = np.zeros((1,num_test_samples))
for j in range(len(h_solution.T)):
    ti,xj = 1/8,tx_eqn1[...,1][j]
    hij = h.subs(t,ti).subs(x,xj)
    c1 = (t,x)
    hij_f = lambdify(c1, hij, modules='numpy')/432
    h_solution[0][j] = hij_f(ti,xj)

hc = h_solution[:,0]