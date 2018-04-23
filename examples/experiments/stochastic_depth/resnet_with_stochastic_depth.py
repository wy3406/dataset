"""
Xie S. et al. "`Aggregated Residual Transformations for Deep Neural Networks
<https://arxiv.org/abs/1611.05431>`_"
"""
import numpy as np
import tensorflow as tf

from dataset.models.tf import ResNet50
from dataset.models.tf.layers import conv_block

class StochasticResNet(ResNet50):
    """ Depended on ResNet50 class with stochastic depth
    """
    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : list of int
            number of filters in each block group
        num_blocks : list of int
            number of blocks in each group
        bottleneck : bool
            whether to use a simple or bottleneck block
        bottleneck_factor : int
            filter number multiplier for a bottleneck block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        filters, block_args = cls.pop(['filters', 'block'], kwargs)
        block_args = {**kwargs, **block_args}
        prob = np.linspace(1, .6, sum(kwargs['num_blocks']))
        global_block = 0
        with tf.variable_scope(name):
            x, inputs = inputs, None
            for i, n_blocks in enumerate(kwargs['num_blocks']):
                with tf.variable_scope('group-%d' % i):
                    for block in range(n_blocks):
                        downsample = i > 0 and block == 0
                        block_args['downsample'] = downsample
                        off = tf.cond(kwargs['is_training'], \
                                      lambda: tf.where(tf.random_uniform([1], 0, 1) > (1 - prob[global_block]),
                                                       tf.ones([1]), tf.zeros([1])),
                                      lambda: tf.ones([1]) * prob[global_block])[0]
                        x = cls.block(x, filters=filters[i], name='block-%d' % block, off=off, **block_args)
                        global_block += 1
                    x = tf.identity(x, name='output')
        return x

    @classmethod
    def block(cls, inputs, name='block', off=1., **kwargs):
        """ A network building block

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        resnext : bool
            whether to use a usuall or aggregated ResNeXt block
        resnext_factor : int
            cardinality for ResNeXt block
        bottleneck : bool
            whether to use a simple or bottleneck block
        bottleneck_factor : int
            the filters nultiplier in the bottleneck block
        se_block : bool
            whether to include squeeze and excitation block
        se_factor : int
            se block ratio
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/block', **kwargs)
        filters, downsample = cls.pop(['filters', 'downsample'], kwargs)
        width_factor = cls.pop('width_factor', kwargs)
        bottleneck, bottleneck_factor = cls.pop(['bottleneck', 'bottleneck_factor'], kwargs)
        resnext, resnext_factor = cls.pop(['resnext', 'resnext_factor'], kwargs)
        se_block, se_factor = cls.pop(['se_block', 'se_factor'], kwargs)
        post_activation = cls.pop('post_activation', kwargs)


        with tf.variable_scope(name):
            filters = filters * width_factor
            if resnext:
                x = cls.next_sub_block(inputs, filters, bottleneck, resnext_factor, name='sub',
                                       downsample=downsample, **kwargs)
            else:
                x = cls.sub_block(inputs, filters, bottleneck, bottleneck_factor, name='sub',
                                  downsample=downsample, **kwargs)

            data_format = kwargs.get('data_format')
            inputs_channels = cls.num_channels(inputs, data_format)
            x_channels = cls.num_channels(x, data_format)

            x = tf.cond(tf.cast(off, tf.bool), lambda: x*off, lambda: tf.zeros_like(x))

            if inputs_channels != x_channels or downsample:
                strides = 2 if downsample else 1
                shortcut = conv_block(inputs, name='shortcut', **{**kwargs, **dict(layout='c', filters=x_channels,
                                                                                   kernel_size=1, strides=strides)})
            else:
                shortcut = inputs

            if se_block:
                x = cls.se_block(x, se_factor, **kwargs)

            x = x + shortcut

            if post_activation:
                x = post_activation(x)

            x = tf.identity(x, name='output')

        return x
