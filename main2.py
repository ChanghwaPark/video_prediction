import os
from pprint import pprint

import tensorflow as tf
from termcolor import colored

from model2 import model
from train2 import train

# Define flag arguments
flags = tf.app.flags

## Data
flags.DEFINE_integer('bs', 64, '80 Batch size')
flags.DEFINE_integer('sz', 64, 'Experiment input size')
flags.DEFINE_integer('ch', 3, 'Experiment number of channels')

## Architecture
flags.DEFINE_string('logdir', 'results/log', 'Log directory')
flags.DEFINE_string('ckptdir', 'results/checkpoints', 'Checkpoint directory')
flags.DEFINE_string('datadir', 'Data', 'Directory for datasets')
flags.DEFINE_string('da', 'none', 'Discriminator activation; tanh, sigmoid')
flags.DEFINE_integer('clip', 0, 'Discriminator weight clipping flag; 0: no clipping, 1: yes clipping')

## Hyper-parameters
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epoch', 320, 'Number of epochs')

## Local GAN hyper-parameters
flags.DEFINE_integer('ngf', 16, 'LGAN generator layers depth')
flags.DEFINE_integer('ndf', 16, 'LGAN discriminator layers depth')
flags.DEFINE_integer('nz', 16, '16 Dimension of latent z vector')
flags.DEFINE_integer('jcb', 8, 'Dimension of jacobian')
flags.DEFINE_float('lrD', 5e-5, '5e-5, 1e-4, Discriminator learning rate')
flags.DEFINE_float('lrG', 1e-3, '1e-3, 5e-4, Generator learning rate')
flags.DEFINE_float('lrDecay', 0.95, 'Decay rate for the learning rate')
flags.DEFINE_float('alpha', 20, 'Weight for locality')
flags.DEFINE_float('beta', 1e-2, 'Weight for orthogonality')
flags.DEFINE_float('theta', 0.1, 'Weight to enforce locality at z = 0')
flags.DEFINE_float('delta', 1e-4, 'Jacobian step size')
flags.DEFINE_float('var', 3, 'Gaussian noise variance for the training')

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
            f"lrD_{FLAGS.lrD}",
            f"lrG_{FLAGS.lrG}",
            f"var_{FLAGS.var}",
            f"phase_{FLAGS.phase}",
            f"da_{FLAGS.da}",
            f"clip_{FLAGS.clip}"
        ]
    else:
        setup_list = [
            f"ngf_{FLAGS.ngf}",
            f"ndf_{FLAGS.ndf}",
            f"nz_{FLAGS.nz}",
            f"lrD_{FLAGS.lrD}",
            f"lrG_{FLAGS.lrG}",
            f"var_{FLAGS.var}",
            f"phase_{FLAGS.phase}",
            f"da_{FLAGS.da}",
            f"clip_{FLAGS.clip}",
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
            f"lrD_{FLAGS.lrD}",
            f"lrG_{FLAGS.lrG}",
            f"var_{FLAGS.var}",
            f"phase_0",
            f"da_{FLAGS.da}",
            f"clip_{FLAGS.clip}"
        ]
        lgan_name = '_'.join(setup_list)
        # just for now
        # lgan_name = 'ngf_64_ndf_64_nz_64_lrD_5e-05_lrG_0.001_dg_1_aug_0_lw_20.0_ow_0.01_var_3.0_phase_0_nosig'
        # lgan_name = 'ngf_64_ndf_64_nz_16_lw_20.0_ow_0.01_var_3.0_phase_0'
        var_lgan = tf.get_collection('trainable_variables', 'lgan/gen')
        path = tf.train.latest_checkpoint(os.path.join(FLAGS.ckptdir, lgan_name))
        tf.train.Saver(var_lgan).restore(M.sess, path)
        print(colored(f"LGAN model is restored from {path}", "blue"))

    saver = tf.train.Saver()

    # Train the main model
    train(M, FLAGS, saver=saver, model_name=model_name)


if __name__ == '__main__':
    tf.app.run()
