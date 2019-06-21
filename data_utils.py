import os
import random

import numpy as np
from PIL import Image


def get_train_batch(batch_size, n_seq=10000, T=10, K=10, dir='Data/train_sequence/'):
    selected_idx = random.sample(range(n_seq), batch_size)
    input = np.zeros([batch_size, T, 64, 64, 3])
    output = np.zeros([batch_size, K, 64, 64, 3])
    for i, idx in enumerate(selected_idx):
        for t in range(T + K):
            img_path = os.path.join(dir, 'sequence%04d' % idx, 'frames%02d.png' % t)
            # img = np.array(Image.open(img_path)) / 127.5 - 1  # normalize
            img = np.array(Image.open(img_path)) / 255.0  # normalize

            if t < 10:
                input[i, t] = img
            else:
                output[i, t - 10] = img

    return input, output


def get_val_batch(start, end, T=10, K=10, dir='Data/val_sequence/'):
    input = np.zeros([end - start, T, 64, 64, 3])
    output = np.zeros([end - start, K, 64, 64, 3])
    for i, idx in enumerate(range(start, end)):
        for t in range(T + K):
            img_path = os.path.join(dir, 'sequence%03d' % idx, 'frames%02d.png' % t)
            # img = np.array(Image.open(img_path)) / 127.5 - 1  # normalize
            img = np.array(Image.open(img_path)) / 255.0  # normalize
            if t < 10:
                input[i, t] = img
            else:
                output[i, t - 10] = img

    return input, output


def get_test_batch(start, end, T=10, dir='Data/test_sequence/'):
    input = np.zeros([end - start, T, 64, 64, 3])
    for i, idx in enumerate(range(start, end)):
        for t in range(T):
            img_path = os.path.join(dir, 'sequence%03d' % idx, 'frames%02d.png' % t)
            # img = np.array(Image.open(img_path)) / 127.5 - 1  # normalize
            img = np.array(Image.open(img_path)) / 255.0  # normalize
            input[i, t] = img

    return input
