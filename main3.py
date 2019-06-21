import os
from pprint import pprint

import tensorflow as tf
from termcolor import colored

from model3 import model
from train3 import train

# Define flag arguments
flags = tf.app.flags

## Data
flags.DEFINE_integer('bs', 64, 'Batch size')
flags.DEFINE_integer('sz', 64, 'Experiment input size')
flags.DEFINE_integer('ch', 3, 'Experiment number of channels')

## Architecture
flags.DEFINE_string('logdir', 'results/log', 'Log directory')
flags.DEFINE_string('ckptdir', 'results/checkpoints', 'Checkpoint directory')
flags.DEFINE_string('datadir', 'Data', 'Directory for datasets')

## Hyper-parameters
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epoch', 320, 'Number of epochs')

## Local GAN hyper-parameters
flags.DEFINE_integer('ngf', 64, 'LGAN generator layers depth')
flags.DEFINE_integer('ndf', 64, 'LGAN discriminator layers depth')
flags.DEFINE_integer('nz', 64, 'Dimension of latent z vector')
flags.DEFINE_float('lrD', 2e-4, '5e-5, 1e-4, Discriminator learning rate')
flags.DEFINE_float('lrG', 2e-4, '1e-3, 5e-4, Generator learning rate')
flags.DEFINE_float('alpha', 20, 'Weight for locality')
flags.DEFINE_float('lrDecay', 0.95, 'Decay rate for the learning rate')


## LSTM hyper-parameters
flags.DEFINE_integer('nhl', 2, 'The number of LSTM layers')
flags.DEFINE_integer('nhw', 128, 'LSTM layer hidden size')
flags.DEFINE_integer('sbs', 64, 'Sequence batch size')

## Others
flags.DEFINE_string('gpu', '1', 'GPU number')
flags.DEFINE_integer('phase', '0', '0   Phase indicator; 0: Train an autoencoder, 1: Train LSTM')

FLAGS = flags.FLAGS


def main(_):
    # Print FLAGS values
    pprint(FLAGS.flag_values_dict())

    # Define GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    # Define model name
    if not FLAGS.phase:
        setup_list = [
            f"ngf_{FLAGS.ngf}",
            f"ndf_{FLAGS.ndf}",
            f"nz_{FLAGS.nz}",
            f"alpha_{FLAGS.alpha}",
            f"phase_{FLAGS.phase}"
        ]
    else:
        setup_list = [
            f"ngf_{FLAGS.ngf}",
            f"ndf_{FLAGS.ndf}",
            f"nz_{FLAGS.nz}",
            f"alpha_{FLAGS.alpha}",
            f"phase_{FLAGS.phase}",
            f"nhl_{FLAGS.nhl}",
            f"nhw_{FLAGS.nhw}"
        ]

    model_name = '_'.join(setup_list)
    print(f"Model name: {model_name}")

    M = model(FLAGS, gpu_config)
    M.sess.run(tf.global_variables_initializer())

    if FLAGS.phase:
        # Previously learned autoencoder model name
        setup_list = [
            f"ngf_{FLAGS.ngf}",
            f"ndf_{FLAGS.ndf}",
            f"nz_{FLAGS.nz}",
            f"alpha_{FLAGS.alpha}",
            f"phase_0"
        ]
        lgan_name = '_'.join(setup_list)
        var_lgan = tf.get_collection('trainable_variables', 'lgan/gen')
        path = tf.train.latest_checkpoint(os.path.join(FLAGS.ckptdir, lgan_name))
        tf.train.Saver(var_lgan).restore(M.sess, path)
        print(colored(f"LGAN model is restored from {path}", "blue"))

    saver = tf.train.Saver()

    # Train the main model
    train(M, FLAGS, saver=saver, model_name=model_name)


if __name__ == '__main__':
    tf.app.run()
