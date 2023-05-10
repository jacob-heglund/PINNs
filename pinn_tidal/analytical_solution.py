import tensorflow as tf

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


def ht0(txini, params):
    """
    system state at t = 0. Not used during training or evaluation of models.
    """
    x = txini[..., 1, None]
    p = 1/(2**0.5)*(((params[8]/params[4])**2+(params[6]*params[3]/params[4])**2)**0.5+params[8]/params[4])**0.5
    z1 = params[6]*params[3]/2/p/params[4]*x+params[7]
    z2 = -p*x
    return tf.cos(z1)*tf.exp(z2)*params[5]+tf.ones_like(x)*params[2]

