import tensorflow as tf
import numpy as np
import scipy.optimize

"""
physical parameters for the tidal system

params[0] = xm, distance from the land
params[1] = tm, time
params[2] = hz, initial head at t=0
params[3] = S, storativity, a dimensionless measure of the volume of water that will be discharged from an aquifer per unit area of the aquifer and per unit reduction in hydraulic head
params[4] = T, transmissivity
params[5] = A, amplitude of tidal changes
params[6] = a, angular velocity
params[7] = c, phase shift of tidal changes
params[8] = L, leakance / specific leakage
"""


class Network:
    """
    Build a physics informed neural network (PINN) model for the wave equation.
    """

    @classmethod
    def build(cls, num_inputs=2, layers=[32, 16, 16, 32], activation='sigmoid', num_outputs=1):
        """
        Build a PINN model for the wave equation with input shape (t, x) and output shape u(t, x).
        Args:
            num_inputs: number of input variables. Default is 2 for (t, x).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 1 for u(t, x).
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
        network: keras network model with input (t, x) and output u(t, x).
        c: wave velocity.
        grads: gradient layer.
    """

    def __init__(self, network, params):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
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
                output: [ u(t,x) relative to equation,
                          u(t=0, x) relative to initial condition,
                          du_dt(t=0, x) relative to initial derivative of t,
                          u(t, x=bounds) relative to boundary condition ]
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
        h_eqn = self.params[3] * dh_dt - self.params[4] * d2h_dx2 - self.params[8] * (-h)
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
