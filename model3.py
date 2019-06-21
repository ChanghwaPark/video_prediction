import tensorbayes as tb
import tensorflow as tf
from tensorbayes.layers import placeholder, constant
from tensorflow.python.ops.losses.losses_impl import absolute_difference as abs_diff
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_xent
from termcolor import colored

from network3 import network


def model(FLAGS, gpu_config):
    """
    :param FLAGS: Contains the experiment info
    :return: (TensorDict) the model
    """

    print(colored("Model initialization started", "blue"))

    nn = network(FLAGS)
    sz = FLAGS.sz
    ch = FLAGS.ch
    bs = FLAGS.bs
    sbs = FLAGS.sbs

    T = tb.utils.TensorDict(dict(
        sess=tf.Session(config=tb.growth_config()),
        x=placeholder((bs, sz, sz, ch)),
        lrD=placeholder(None),
        lrG=placeholder(None),
        seq_in=placeholder((10, sbs, sz, sz, ch)),
        seq_out=placeholder((10, sbs, sz, sz, ch)),
        val_seq_in=placeholder((10, 10, sz, sz, ch)),
        val_seq_out=placeholder((10, 10, sz, sz, ch)),
        test_seq_in=placeholder((10, 10, sz, sz, ch)),
        lr=placeholder(None)
    ))

    recon_x = nn.generator(T.x, phase=True)

    # Compute discriminator logits
    real_logit = nn.discriminator(T.x, phase=True)
    fake_logit = nn.discriminator(recon_x, phase=True)

    # Adversarial generator
    loss_disc = tf.reduce_mean(
        sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
        sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit))
    loss_fake = tf.reduce_mean(
        sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit))

    loss_local = tf.reduce_mean(abs_diff(labels=T.x, predictions=recon_x))

    loss_gen = loss_fake + FLAGS.alpha * loss_local

    var_gen = tf.get_collection('trainable_variables', 'lgan/gen')
    train_gen = tf.train.AdamOptimizer(T.lrG, 0.5).minimize(loss_gen, var_list=var_gen)

    var_disc = tf.get_collection('trainable_variables', 'lgan/dsc')
    train_disc = tf.train.AdamOptimizer(T.lrD, 0.5).minimize(loss_disc, var_list=var_disc)

    # Summarizations
    summary_disc = [tf.summary.scalar('disc/loss_disc', loss_disc)]
    summary_gen = [tf.summary.scalar('gen/loss_gen', loss_gen),
                   tf.summary.scalar('gen/loss_local', loss_local),
                   tf.summary.scalar('gen/loss_fake', loss_fake),
                   tf.summary.scalar('hyper/lrD', T.lrD),
                   tf.summary.scalar('hyper/lrG', T.lrG)]
    summary_image = [tf.summary.image('image/x', T.x),
                     tf.summary.image('image/recon_x', recon_x)]

    # Merge summaries
    summary_disc = tf.summary.merge(summary_disc)
    summary_gen = tf.summary.merge(summary_gen)
    summary_image = tf.summary.merge(summary_image)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('disc'), loss_disc,
                   c('gen'), loss_gen,
                   c('local'), loss_local,
                   c('fake'), loss_fake]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_gen = [summary_gen, train_gen]
    T.ops_image = summary_image

    if FLAGS.phase:
        # LSTM initialization
        seq_in = tf.reshape(T.seq_in, [-1, sz, sz, ch])
        seq_out = tf.reshape(T.seq_out, [-1, sz, sz, ch])
        val_seq_in = tf.reshape(T.val_seq_in, [-1, sz, sz, ch])
        test_seq_in = tf.reshape(T.test_seq_in, [-1, sz, sz, ch])
        enc_in = nn.generator(seq_in, phase=True, enc=True)
        enc_out = nn.generator(seq_out, phase=True, enc=True)
        val_enc_in = nn.generator(val_seq_in, phase=True, enc=True)
        test_enc_in = nn.generator(test_seq_in, phase=True, enc=True)
        enc_in = tf.stop_gradient(enc_in)
        enc_out = tf.stop_gradient(enc_out)
        val_enc_in = tf.stop_gradient(val_enc_in)
        test_enc_in = tf.stop_gradient(test_enc_in)
        enc_in = tf.squeeze(enc_in)
        enc_out = tf.squeeze(enc_out)
        val_enc_in = tf.squeeze(val_enc_in)
        test_enc_in = tf.squeeze(test_enc_in)
        enc_in = tf.reshape(enc_in, [-1, sbs, FLAGS.nz])
        enc_out = tf.reshape(enc_out, [-1, sbs, FLAGS.nz])
        val_enc_in = tf.reshape(val_enc_in, [-1, 10, FLAGS.nz])
        test_enc_in = tf.reshape(test_enc_in, [-1, 10, FLAGS.nz])

        with tf.variable_scope('lstm/in'):
            in_cell = tf.contrib.cudnn_rnn.CudnnLSTM(FLAGS.nhl, FLAGS.nhw, dropout=0.5)

            _, in_states = in_cell(enc_in, initial_state=None, training=True)
            _, val_in_states = in_cell(val_enc_in, initial_state=None, training=False)
            _, test_in_states = in_cell(test_enc_in, initial_state=None, training=False)

        with tf.variable_scope('lstm/out'):
            out_cell = tf.contrib.cudnn_rnn.CudnnLSTM(FLAGS.nhl, FLAGS.nhw, dropout=0.5)

            outputs, _ = out_cell(tf.zeros_like(enc_out), initial_state=in_states, training=True)
            val_outputs, _ = out_cell(tf.zeros_like(val_enc_in), initial_state=val_in_states, training=False)
            test_outputs, _ = out_cell(tf.zeros_like(test_enc_in), initial_state=test_in_states, training=False)

            enc_out_pred = tf.layers.dense(outputs, FLAGS.nz, activation=None, name='lstm_dense', reuse=tf.AUTO_REUSE)
            val_enc_out_pred = tf.layers.dense(val_outputs, FLAGS.nz, activation=None, name='lstm_dense',
                                               reuse=tf.AUTO_REUSE)
            test_enc_out_pred = tf.layers.dense(test_outputs, FLAGS.nz, activation=None, name='lstm_dense',
                                               reuse=tf.AUTO_REUSE)

        enc_out_pred_reshape = tf.reshape(enc_out_pred, [-1, FLAGS.nz])
        enc_out_pred_reshape = tf.expand_dims(tf.expand_dims(enc_out_pred_reshape, 1), 1)
        val_enc_out_pred_reshape = tf.reshape(val_enc_out_pred, [-1, FLAGS.nz])
        val_enc_out_pred_reshape = tf.expand_dims(tf.expand_dims(val_enc_out_pred_reshape, 1), 1)
        test_enc_out_pred_reshape = tf.reshape(test_enc_out_pred, [-1, FLAGS.nz])
        test_enc_out_pred_reshape = tf.expand_dims(tf.expand_dims(test_enc_out_pred_reshape, 1), 1)

        seq_out_pred = nn.generator(enc_out_pred_reshape, phase=True, dec=True)
        seq_out_pred = tf.reshape(seq_out_pred, [10, sbs, sz, sz, ch])
        val_seq_out_pred = nn.generator(val_enc_out_pred_reshape, phase=True, dec=True)
        val_seq_out_pred = tf.reshape(val_seq_out_pred, [10, 10, sz, sz, ch])
        test_seq_out_pred = nn.generator(test_enc_out_pred_reshape, phase=True, dec=True)
        T.test_seq_out_pred = tf.reshape(test_seq_out_pred, [10, 10, sz, sz, ch])

        T.val_mae = tf.reduce_mean(abs_diff(labels=T.val_seq_out, predictions=val_seq_out_pred))
        loss_lstm = tf.reduce_mean(abs_diff(labels=enc_out, predictions=enc_out_pred))
        var_lstm = tf.get_collection('trainable_variables', 'lstm')
        # train_lstm = tf.train.AdamOptimizer(FLAGS.lr, 0.5).minimize(loss_lstm, var_list=var_lstm)
        train_lstm = tf.train.AdamOptimizer(T.lr, 0.5).minimize(loss_lstm, var_list=var_lstm)

        summary_lstm = [tf.summary.scalar('lstm/loss_lstm', loss_lstm)]
        summary_lstm_image = [tf.summary.image('lstm/seq_out', T.seq_out[:, 0, :, :, :]),
                              tf.summary.image('lstm/seq_out_pred', seq_out_pred[:, 0, :, :, :])]
        summary_lstm = tf.summary.merge(summary_lstm)
        summary_lstm_image = tf.summary.merge(summary_lstm_image)

        T.ops_lstm_print = [c('loss_lstm'), loss_lstm]
        T.ops_lstm = [summary_lstm, train_lstm]
        T.ops_lstm_image = summary_lstm_image

    print(colored("Model initialization ended", "blue"))

    return T
