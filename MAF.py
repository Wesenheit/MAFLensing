import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfk=tf.keras
tfkl=tf.keras.layers
tfd=tfp.distributions
tfb=tfp.bijectors


def MAF(self,ndim,conddim,num,dis,hidden_units,reverse=True,addnorm=False,activation="relu"):
    """
    ndim - number of dimensions we want to model
    conddim - number of dimensions of conditional vector
    num - number of made block
    dis - base distribution, number of dimensions of dis must be equal to ndim
    hidden_units - list of hiden units for made block
    reverse - if we want to reverse order affter each block
    addnorm - if we want to add batch normalisation layer
    activation - type of activaion
    """
    bijectors=[]
    for _ in range(num):
        made = tfb.AutoregressiveNetwork(params=ndim, event_shape=[ndim,], hidden_units=hidden_units, activation=activation)
        bijectors.append(tfb.MaskedAutoregressivFlov(shift_and_log_scale_fn=made))
        if reverse:
            bijectors.append(tfb.Permute(permutation=[ndim-i for i in range(ndim)]))
        if addnorm:
            bijectors.append(tfb.BatchNormalization())
    flow_bijector = tfb.Chain(bijectors)
    tf_dist=tfd.TransformedDistribution(distribution=dis, bijector=flow_bijector)
    input_x=tfkl.Input(shape=(ndim,), dtype=tf.float32)
    input_cond=tfkl.Input(shape=(conddim,), dtype=tf.float32)
    prob = tf_dist.prob(input_x, bijector_kwargs={'conditional_input': input_cond})
    return tfk.Model([input_x, input_cond], prob)





