import tensorflow as tf
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import scipy.stats.qmc as qmc
tf.random.set_seed(12321)

class Network:
    """
    Build a physics informed neural network (PINN) model for the wave equation.
    """

    @classmethod
    def build(cls, num_inputs=2, layers=[32, 16, 16, 32], activation='sigmoid', num_outputs=1):
        """
        Build a PINN model for the wave equation with input shape (t, x) and output shape h(t, x).
        Args:
            num_inputs: number of input variables. Default is 2 for (t, x).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 1 for h(t, x).
        Returns:
            keras network model.
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                kernel_initializer='glorot_uniform')(x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='glorot_uniform')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for the wave equation.
    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, tx):
        """
        Computing 1st and 2nd derivatives for the wave equation.
        Args:
            tx: input variables (t, x).
        Returns:
            u: network output.
            du_dt: 1st derivative of t.
            du_dx: 1st derivative of x.
            d2u_dt2: 2nd derivative of t.
            d2u_dx2: 2nd derivative of x.
        """

        with tf.GradientTape() as g:
            g.watch(tx)
            with tf.GradientTape() as gg:
                gg.watch(tx)
                h = self.model(tx)
            dh_dtx = gg.batch_jacobian(h, tx)
            dh_dt = dh_dtx[..., 0]
            dh_dx = dh_dtx[..., 1]
        d2h_dtx2 = g.batch_jacobian(dh_dtx, tx)
        d2h_dt2 = d2h_dtx2[..., 0, 0]
        d2h_dx2 = d2h_dtx2[..., 1, 1]

        return h, dh_dt, dh_dx, d2h_dt2, d2h_dx2

class PINN:
    """
    Build a physics informed neural network (PINN) model for the wave equation.
    Attributes:
        network: keras network model with input (t, x) and output h(t, x).
        c: wave velocity.
        grads: gradient layer.
    """

    def __init__(self, network, params):
        """
        Args:
            network: keras network model with input (t, x) and output h(t, x).
            c: wave velocity. Default is 1.
        """

        self.network = network
        self.params = params
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the wave equation.
        Returns:
            PINN model for the projectile motion with
                input: [ (t, x) relative to equation,
                         (t=0, x) relative to initial condition,
                         (t, x=bounds) relative to boundary condition ],
                output: [ h(t,x) relative to equation,
                          h(t=0, x) relative to initial condition,
                          du_dt(t=0, x) relative to initial derivative of t,
                          h(t, x=bounds) relative to boundary condition ]
        """

        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        # initial condition input:(0,x)
        t_ini = tf.keras.layers.Input(shape=(2,))
        # boundary condition 1 input: (t, x=0)
        tx_bnd1 = tf.keras.layers.Input(shape=(2,))
        # boundary condition 2 input: (t, x=3000)
        tx_bnd2 = tf.keras.layers.Input(shape=(2,))

        # compute gradients
        h, dh_dt, _, _, d2h_dx2 = self.grads(tx_eqn)

        # equation residual
        h_eqn = self.params[3]*dh_dt-self.params[4]*d2h_dx2-self.params[8]*(-h)
        #initial condition output
        h_ini = self.network(t_ini)
        # boundary condition 1 output
        h_b1= self.network(tx_bnd1)
        # boundary condition 2 residual
        h_b2 = self.network(tx_bnd2)# dirichlet
        #_, _, u_bnd, _, _ = self.grads(tx_bnd)  # neumann

        # build the PINN model for the wave equation
        return tf.keras.models.Model(
            inputs=[tx_eqn, tx_bnd1, tx_bnd2],
            outputs=[h_eqn, h_b1, h_b2])
#params=[xm,tm,hz,S,T,A,a,c,L]
#params=[0,  1, 2,3,4,5,6,7,8]
class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.
    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
               1e7 for moderate accuracy; 10 for extremely high accuracy.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metris: logging metrics.
        progbar: progress bar.
    """

    def __init__(self, model, x_train, y_train, factr=10, m=50, maxls=50, maxiter=20000):
        """
        Args:
            model: optimization target model.
            samples: training samples.
            factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
                   1e7 for moderate accuracy; 10.0 for extremely high accuracy.
            m: maximum number of variable metric corrections used to define the limited memory matrix.
            maxls: maximum number of line search steps (per iteration).
            maxiter: maximum number of iterations.
        """

        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        # initialize the progress bar
        self.progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params( {
            'verbose':1, 'epochs':1, 'steps':self.maxiter, 'metrics':self.metrics})

    def set_weights(self, flat_weights):
        """
        Set weights to the model.
        Args:
            flat_weights: flatten weights.
        """

        # get model weights
        shapes = [ w.shape for w in self.model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Evaluate loss and gradients for weights as tf.Tensor.
        Args:
            x: input data.
        Returns:
            loss and gradients for weights as tf.Tensor.
        """

        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), y))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as ndarray.
        Args:
            weights: flatten weights.
        Returns:
            loss and gradients for weights as ndarray.
        """

        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def callback(self, weights):
        """
        Callback that prints the progress to stdout.
        Args:
            weights: flatten weights.
        """
        self.progbar.on_batch_begin(0)
        loss, _ = self.evaluate(weights)
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))

    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """

        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.model.get_weights() ])
        # optimize the weight vector
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
            factr=self.factr, m=self.m, maxls=self.maxls, maxiter=self.maxiter,
            callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()
#params=[xm,tm,hz,S,T,A,a,c,L]
#params=[0,  1, 2,3,4,5,6,7,8]

class Adam(tf.keras.Model):
        def __init__(self,model):
            super(Adam,self).__init__()
            self.model = model


        def train_step(self,data):
            x,y=data
            with tf.GradientTape() as tape:

                y_pred = self.model(x,training=True)


                loss = self.compiled_loss(y,y_pred,regularization_losses=self.losses)


            training_vars = self.trainable_variables
            gradients= tape.gradient(loss,training_vars)


            self.optimizer.apply_gradients(zip(gradients,training_vars))


            return {m.name: m.result() for m in self.metrics}








def h0(txbnd1, params):
    """
    boundary wave form.
    Args:
        txbnd1: variables at boundary condition 1 (t, x=0) as tf.Tensor.
        a: angular velocity. params[6]
        c: phase shift. params[7]

    Returns:
        h(t, 0) as tf.Tensor.
    """

    t = txbnd1[..., 0, None]
    z = params[6]*t+params[7]
    return tf.cos(z) *params[5]+tf.ones_like(t)*params[2]
#params=[xm,tm,hz,S,T,A,a,c,L]
#params=[0,  1, 2,3,4,5,6,7,8]
def ht0(txini,params):

    x = txini[..., 1, None]
    p = 1/(2**0.5)*(((params[8]/params[4])**2+(params[6]*params[3]/params[4])**2)**0.5+params[8]/params[4])**0.5
    z1 = params[6]*params[3]/2/p/params[4]*x+params[7]
    z2 = -p*x
    return tf.cos(z1)*tf.exp(z2)*params[5]+tf.ones_like(x)*params[2]

params= [2000,48,0,0.001,2000/24,0.65,2*np.pi/24,0,0,24,96]
if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for the wave equation.
    """

    # number of training samples
    num_train_samples = 240000
    # number of test samples
    num_test_samples = 120000
    n1 =2*num_train_samples
    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network,params).build()

    # create training input
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=num_train_samples)
    qmc.discrepancy(sample)


    tx_eqn = np.zeros((num_train_samples,2))
    tx_eqn[...,0]=np.random.uniform(0, params[1], num_train_samples)
    tx_eqn[...,1]=np.random.uniform(0, params[0], num_train_samples)

    tx_ini = np.zeros((num_train_samples,2))
    tx_ini[...,1]=np.random.uniform(0, params[0], num_train_samples)
                                                 # x = 0 ~ +3000
    tx_bnd1 = np.zeros((num_train_samples,2))
    tx_bnd1[..., 0] = np.random.uniform(0, params[1], num_train_samples)             # t = 0 ~ +24
                                                    # x = 0
    tx_bnd2 = np.ones((num_train_samples,2))
    tx_bnd2[..., 0] = np.random.uniform(0, params[1], num_train_samples)             # t =  0 ~ +4
    tx_bnd2[..., 1] = params[0]*tx_bnd2[..., 1]  # x = 00
    # create training output
    h_zero = np.zeros((num_train_samples, 1))
    h_ini = ht0(tf.constant(tx_ini),params).numpy()
    h_bnd1 = h0(tf.constant(tx_bnd1),params).numpy()

    # create training data
    x_train = [tx_eqn, tx_bnd1, tx_bnd2]
    y_train = [h_zero, h_bnd1, h_zero]

    # L-BFGS-B
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # Adam
    Adam_optimizer = Adam(pinn)


    Adam_optimizer.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3,
                                               beta_1 = 0.9, beta_2 = 0.999,
                                               epsilon = 1e-08, name = 'Adam'),
                   loss = tf.keras.losses.MeanSquaredError(),metrics=["mae"])



    Adam_optimizer.fit(x_train,y_train,epochs = 7)
    num_test_samples_t = num_test_samples
    # predict h(t,x) distribution
    t_flat0 = np.linspace(0, (params[9]/2)-1, int(num_test_samples_t/(params[10]/params[9]*2)))
    ho=int(params[10]/params[9]*2)
    t_flat1 = np.tile(t_flat0,int(params[10]/params[9]*2)).flatten()
    t_flat =  np.zeros_like(t_flat1)
    for i in range(len(t_flat)):
        if i <int(num_test_samples_t/(params[10]/params[9]*2)):
            t_flat[i]=t_flat1[i]
        else:
            t_flat[i]=round(t_flat1[i]+12*(i//(len(t_flat0))),5)
    x_flat = np.linspace(0, params[0], num_test_samples)



    # plot h(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    # plot h(t=const, x) cross-sections
    t_cross_sections = [12, 18,24]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        if 0<=t_cs<params[9]/2:
            tx1 = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
            h1 = network.predict(tx1, batch_size=num_test_samples)
        else:
            tx1 = np.stack([np.full(t_flat.shape, t_cs%(params[9]/2)), x_flat], axis=-1)
            h1 = (-1)**(t_cs//(params[9]/2))*network.predict(tx1,batch_size=num_test_samples)
        plt.plot(x_flat, h1)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('h(t,x)')
    plt.tight_layout()
    plt.savefig('result_unconfined.png', transparent=True)
    plt.show()

    x_cross_sections = [1500,1800,2000]

    for i, x_cs in enumerate(x_cross_sections):
        plt.subplot(gs[1, i])
        d = params[1]/num_test_samples
        tf1 = np.arange(0,params[9]/2,d)
        tf2 = np.tile(tf1,int(2*d*num_test_samples/params[9]))
        txp = np.stack([tf2,np.full(tf2.shape, x_cs)],axis=-1)
        hr = network.predict(txp, batch_size=len(tf2))
        ht = hr.flatten()
        hn=ht
        for j in range(len(hn)):
            if (j % (int(params[9]/2/d)) == 0) & ((-1)**(j//int(params[9]/2/d))<0):


                hn[j:j+int(params[9]/2/d)]=ht[j:j+int(params[9]/2/d)][::-1]
            else:
                hn[j]=ht[j]

        t_flat2 = tf2
        for k in range(len(t_flat2)):
            t_flat2[k]=tf2[k]+params[9]/2*(k//(int(params[9]/2/d)))


        plt.plot(t_flat2.flatten(), hn.flatten())
        plt.title('x={}'.format(x_cs))
        plt.xlabel('t')
        plt.ylabel('h(t,x)')
    plt.tight_layout()
    plt.savefig('result_unconfined1.png', transparent=True)
    plt.show()
    # compute parameter p
    p = 1/(2**0.5)*(((params[8]/params[4])**2+(params[6]*params[3]/params[4])**2)**0.5+params[8]/params[4])**0.5
    # analytical solutions
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

