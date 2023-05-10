import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pdb

def create_figures(network, params, eval_params, n_test=120000):

    # predict h(t,x) distribution
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

    # plot h(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])

    # plot h(t=const, x) cross-sections
    t_cross_sections = [6, 12, 24]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        if 0 <= t_cs < eval_params[0] / 2:
            tx1 = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
            h1 = network.predict(tx1, batch_size=n_test)
        else:
            tx1 = np.stack([np.full(t_flat.shape, t_cs %  (eval_params[0] / 2)), x_flat], axis=-1)
            h1 = (-1)**(t_cs // (eval_params[0] / 2)) * network.predict(tx1, batch_size=n_test)

        plt.plot(x_flat, h1)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('h(t,x)')
    plt.tight_layout()
    plt.savefig('result_unconfined.png', transparent=True)
    plt.show()

    # plot h(t, x=const) cross-sections
    x_cross_sections = [1500, 1800, 2000]

    for i, x_cs in enumerate(x_cross_sections):
        plt.subplot(gs[1, i])
        d = params[1] / n_test
        tf1 = np.arange(0, eval_params[0] / 2, d)
        tf2 = np.tile(tf1, int(2 * d * n_test / eval_params[0]))
        txp = np.stack([tf2, np.full(tf2.shape, x_cs)], axis=-1)
        hr = network.predict(txp, batch_size=len(tf2))
        ht = hr.flatten()
        hn = ht

        #TODO what is this value? Something to do with time indexing?
        magic_value = int(eval_params[0] / 2 / d)

        for j in range(len(hn)):
            if (j % magic_value == 0) & ((-1)**(j // magic_value) < 0):
                hn[j : j + magic_value] = ht[j : j + magic_value][::-1]
            else:
                hn[j] = ht[j]

        t_flat2 = tf2

        for k in range(len(t_flat2)):
            t_flat2[k] = tf2[k] + eval_params[0] / 2 * (k // magic_value)


        plt.plot(t_flat2.flatten(), hn.flatten())
        plt.title('x={}'.format(x_cs))
        plt.xlabel('t')
        plt.ylabel('h(t,x)')

    plt.tight_layout()
    plt.savefig('result_unconfined1.png', transparent=True)
    plt.show()

    # analytical solutions
    # compute parameter p
    p = 1/(2**0.5)*(((params[8]/params[4])**2+(params[6]*params[3]/params[4])**2)**0.5+params[8]/params[4])**0.5
    t3 = np.linspace(0,params[1],params[1])
    x1 = np.linspace(0,params[0],params[0])
    h3 =  np.ones_like(x1)* params[2]+ params[5]*np.exp(-p*x1)*np.cos(params[6]*12-params[6]*params[3]/2/p/params[4]*x1+params[7])
    h4 =  np.ones_like(t3)* params[2]+ params[5]*np.exp(-p*200)*np.cos(params[6]*t3-params[6]*params[3]/2/p/params[4]*200+params[7])
    plt.plot(x1,h3)
    plt.plot(t3,h4)

    plt.plot(x1,h3,label='real')
    plt.plot(x_flat,h1,label='pred')
    plt.legend()

    plt.plot(t3,h4,label='real')
    plt.plot(t_flat2.flatten(),hn.flatten(),label='pred')
    plt.legend()