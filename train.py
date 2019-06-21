import os

import random
import numpy as np
import tensorbayes as tb
import tensorflow as tf
from PIL import Image
from termcolor import colored

from data_utils import get_train_batch, get_val_batch, get_test_batch
# from image_preprocess import get_train_batch, get_val_batch, get_test_batch
from utils import delete_existing, save_model


def update_dict(L, feed_dict, FLAGS):
    if not FLAGS.phase:
        data_x = np.zeros([FLAGS.bs, FLAGS.sz, FLAGS.sz, FLAGS.ch])
        # batch_sequences = int(FLAGS.bs / 16)
        # seq_in, seq_out = get_train_batch(batch_sequences)
        # for i in range(batch_sequences):
        #     data_x[(i * 16):(i * 16 + 8)] = seq_in[i, :8]
        #     data_x[(i * 16 + 8):(i * 16 + 16)] = seq_out[i, :8]
        seq_in, seq_out = get_train_batch(FLAGS.bs)
        for i in range(FLAGS.bs):
            j = random.randint(0, 1)
            k = random.randint(0, 9)
            if j:
                data_x[i] = seq_in[i, k]
            else:
                data_x[i] = seq_out[i, k]
        z = np.random.normal(0, FLAGS.var, (FLAGS.bs, FLAGS.nz))
        pos = np.eye(FLAGS.nz)[np.random.permutation(FLAGS.nz)[:FLAGS.jcb]]
        pos = np.tile(np.expand_dims(pos, 0), [FLAGS.bs, 1, 1])
        pos = np.reshape(np.transpose(pos, [1, 0, 2]), [FLAGS.bs * FLAGS.jcb, -1])
        iorth = np.tile(np.expand_dims(np.eye(FLAGS.jcb), 0), [FLAGS.bs, 1, 1])
        feed_dict.update({L.x: data_x, L.z: z, L.pos: pos, L.iorth: iorth})
    else:
        seq_in, seq_out = get_train_batch(FLAGS.sbs)
        seq_in = np.transpose(seq_in, [1, 0, 2, 3, 4])
        seq_out = np.transpose(seq_out, [1, 0, 2, 3, 4])
        feed_dict.update({L.seq_in: seq_in, L.seq_out: seq_out})


def train(L, FLAGS, saver=None, model_name=None):
    """
    :param L: (TensorDict) the model
    :param FLAGS: (FLAGS) contains experiment info
    :param saver: (Saver) saves models during training
    :param model_name: name of the model being run with relevant parms info
    :return: None
    """
    bs = FLAGS.bs
    lrD = FLAGS.lrD
    lrG = FLAGS.lrG
    lr = FLAGS.lr
    iterep = 1000
    itersave = 20000
    n_epoch = FLAGS.epoch
    epoch = 0
    feed_dict = {L.lrD: lrD, L.lrG: lrG, L.lr: lr}

    # Create a log directory and FileWriter
    log_dir = os.path.join(FLAGS.logdir, model_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)

    # Create a save directory
    if saver:
        model_dir = os.path.join(FLAGS.ckptdir, model_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)

    print(f"Batch size: {bs}")
    print(f"Iterep: {iterep}")
    print(f"Total iterations: {n_epoch * iterep}")
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint directory: {model_dir}")

    if not FLAGS.phase:
        print(colored("LGAN Training started.", "blue"))

        for i in range(n_epoch * iterep):
            # for j in range(FLAGS.dg):
            # Train the discriminator
            update_dict(L, feed_dict, FLAGS)
            # summary, _ = L.sess.run(L.ops_disc, feed_dict)

            if FLAGS.clip:
                summary, _, _ = L.sess.run(L.ops_disc, feed_dict)
            else:
                summary, _ = L.sess.run(L.ops_disc, feed_dict)

            # if not FLAGS.wgan:
            #     summary, _ = L.sess.run(L.ops_disc, feed_dict)
            # else:
            #     summary, _, _ = L.sess.run(L.ops_disc, feed_dict)

            train_writer.add_summary(summary, i + 1)

            # Train the generator and the classifier
            update_dict(L, feed_dict, FLAGS)
            summary, _ = L.sess.run(L.ops_gen, feed_dict)
            train_writer.add_summary(summary, i + 1)
            train_writer.flush()

            end_epoch, epoch = tb.utils.progbar(i, iterep,
                                                message='{}/{}'.format(epoch, i),
                                                display=True)

            # if not end_epoch:
            if end_epoch:
                summary = L.sess.run(L.ops_image, feed_dict)
                train_writer.add_summary(summary, i + 1)
                train_writer.flush()

                lrD *= FLAGS.lrDecay
                lrG *= FLAGS.lrDecay
                feed_dict.update({L.lrD: lrD, L.lrG: lrG})
                print_list = L.sess.run(L.ops_print, feed_dict)

                for j, item in enumerate(print_list):
                    if j % 2 == 0:
                        print_list[j] = item.decode("ascii")
                    else:
                        print_list[j] = round(item, 5)

                print_list += ['epoch', epoch]
                print(print_list)

            if saver and (i + 1) % itersave == 0:
                save_model(saver, L, model_dir, i + 1)

        # Saving final model
        if saver:
            save_model(saver, L, model_dir, i + 1)
        print(colored("LGAN Training ended.", "blue"))

    else:
        print(colored("LSTM Training started.", "blue"))
        min_val_mae = 10
        # update_dict(L, feed_dict, FLAGS)
        #
        # test1 = L.sess.run(L.test1, feed_dict)
        # # print(test1.shape)
        # # print(test[0, 0])
        # # print(test[1, 0])
        # plt.imshow(test1[0, 0])
        # plt.show()
        # plt.imshow(test1[1, 0])
        # plt.show()
        for i in range(n_epoch * iterep):
            update_dict(L, feed_dict, FLAGS)
            summary, _ = L.sess.run(L.ops_lstm, feed_dict)
            train_writer.add_summary(summary, i + 1)
            train_writer.flush()

            end_epoch, epoch = tb.utils.progbar(i, iterep,
                                                message='{}/{}'.format(epoch, i),
                                                display=True)

            if end_epoch:
                # print("!")
                # summary = L.sess.run(L.ops_lstm_image, feed_dict)
                # train_writer.add_summary(summary, i + 1)
                # train_writer.flush()
                # print("!!")
                val_mae = 0
                for j in range(50):
                    val_seq_in, val_seq_out = get_val_batch(j * 10, (j + 1) * 10)
                    val_seq_in = np.transpose(val_seq_in, [1, 0, 2, 3, 4])
                    val_seq_out = np.transpose(val_seq_out, [1, 0, 2, 3, 4])
                    feed_dict.update({L.val_seq_in: val_seq_in, L.val_seq_out: val_seq_out})
                    current_val_mae = L.sess.run(L.val_mae, feed_dict)
                    val_mae += current_val_mae
                val_mae = val_mae / 50.0 * 255.0
                print_list = L.sess.run(L.ops_lstm_print, feed_dict)

                for j, item in enumerate(print_list):
                    if j % 2 == 0:
                        print_list[j] = item.decode("ascii")
                    else:
                        print_list[j] = round(item, 5)

                print_list += ['val_mae', val_mae]
                print_list += ['epoch', epoch]
                print(print_list)

                if True and val_mae < min_val_mae:
                    min_val_mae = val_mae
                    for j in range(50):
                        test_seq_in = get_test_batch(j * 10, (j + 1) * 10)
                        test_seq_in = np.transpose(test_seq_in, [1, 0, 2, 3, 4])
                        feed_dict.update({L.test_seq_in: test_seq_in})
                        test_seq_out_pred = L.sess.run(L.test_seq_out_pred, feed_dict)

                        test_seq_out_pred[test_seq_out_pred < 0] = 0
                        test_seq_out_pred = (test_seq_out_pred * 255).astype(np.uint8)

                        for k in range(10):
                            idx = j * 10 + k
                            for t in range(10):
                                if not os.path.exists('test_predicted'):
                                    os.mkdir('test_predicted')
                                folder_name = os.path.join('test_predicted/', 'sequence%03d' % idx)
                                if not os.path.exists(folder_name):
                                    os.mkdir(folder_name)
                                img_path = os.path.join(folder_name, 'frames%02d.png' % t)
                                this_img = Image.fromarray(test_seq_out_pred[t, k])
                                this_img.save(img_path)

                lr *= FLAGS.lrDecay
                feed_dict.update({L.lr: lr})

            if saver and (i + 1) % itersave == 0:
                save_model(saver, L, model_dir, i + 1)

        # Saving final model
        if saver:
            save_model(saver, L, model_dir, i + 1)
        print(colored("LSTM Training ended.", "blue"))
