import tensorflow as tf
from tensorbayes.layers import conv2d, conv2d_transpose, batch_norm
from tensorflow.contrib.framework import arg_scope

from utils import leaky_relu, relu, tanh, sigmoid


class network(object):
    def __init__(self, FLAGS):
        self.ngf = FLAGS.ngf
        self.ndf = FLAGS.ndf
        self.nz = FLAGS.nz
        self.sz = FLAGS.sz
        self.ch = FLAGS.ch
        self.bs = FLAGS.bs
        if FLAGS.da == 'none':
            self.da = None
        elif FLAGS.da == 'sigmoid':
            self.da = sigmoid
        elif FLAGS.da == 'tanh':
            self.da = tanh
        else:
            raise ValueError(f"FLAG.da error: {FLAGS.da}")

    def generator(self, x, z, phase, reuse=tf.AUTO_REUSE, internal_update=False, getter=None, enc=False, dec=False):
        with tf.variable_scope('lgan/gen', reuse=reuse, custom_getter=getter):
            with arg_scope([leaky_relu], a=0.2), \
                 arg_scope([conv2d], activation=leaky_relu, bn=True, phase=phase), \
                 arg_scope([conv2d_transpose], activation=relu, bn=True, phase=phase), \
                 arg_scope([batch_norm], internal_update=internal_update):
                layout = [
                    (conv2d, (self.ngf, 4, 2), dict(bn=False)),
                    (conv2d, (self.ngf * 2, 4, 2), {}),
                    (conv2d, (self.ngf * 4, 4, 2), {}),
                    (conv2d, (self.ngf * 8, 4, 2), {}),
                    (conv2d, (self.nz, 4, 1), dict(activation=None, padding="VALID")),
                    (leaky_relu, (), {}),
                    (conv2d_transpose, (self.ngf * 8, 4, 1),
                     dict(padding="VALID", output_shape=[x.get_shape()[0], 4, 4, self.ngf * 8])),
                    (conv2d_transpose, (self.ngf * 4, 4, 2), {}),
                    (conv2d_transpose, (self.ngf * 2, 4, 2), {}),
                    (conv2d_transpose, (self.ngf, 4, 2), {}),
                    (conv2d_transpose, (self.ch, 4, 2), dict(activation=tanh, bn=False))
                ]

                start = 0
                z_layer = 5
                end = len(layout)

                if not dec:
                    for i in range(start, z_layer):
                        with tf.variable_scope('l{:d}'.format(i)):
                            f, f_args, f_kwargs = layout[i]
                            x = f(x, *f_args, **f_kwargs)
                    if enc:
                        return x
                    x = x + tf.expand_dims(tf.expand_dims(z, 1), 1)

                for i in range(z_layer, end):
                    with tf.variable_scope('l{:d}'.format(i)):
                        f, f_args, f_kwargs = layout[i]
                        x = f(x, *f_args, **f_kwargs)

        return x

    def discriminator(self, x, phase, reuse=tf.AUTO_REUSE, internal_update=False, getter=None):
        with tf.variable_scope('lgan/dsc', reuse=reuse, custom_getter=getter):
            with arg_scope([leaky_relu], a=0.2), \
                 arg_scope([conv2d], activation=leaky_relu, bn=True, phase=phase), \
                 arg_scope([batch_norm], internal_update=internal_update):
                layout = [
                    (conv2d, (self.ndf, 4, 2), dict(bn=False)),
                    (conv2d, (self.ndf * 2, 4, 2), {}),
                    (conv2d, (self.ndf * 4, 4, 2), {}),
                    (conv2d, (self.ndf * 8, 4, 2), {}),
                    (conv2d, (1, 4, 1), dict(activation=self.da, bn=False, padding='VALID'))
                    # Loss terms also use sigmoid
                    # (conv2d, (1, 4, 1), dict(activation=None, bn=False, padding='VALID'))
                ]

                for i in range(0, len(layout)):
                    with tf.variable_scope('l{:d}'.format(i)):
                        f, f_args, f_kwargs = layout[i]
                        x = f(x, *f_args, **f_kwargs)

        return x
