import os
import random

import numpy as np
from PIL import Image

# dir = 'Data/train_sequence/'
# for i in range(10000):
#     if not os.path.exists(os.path.join(dir, 'sequence%04d' % i, 'seq.npy')):
#         seq_img = np.zeros([20, 64, 64, 3])
#         for t in range(20):
#             img_path = os.path.join(dir, 'sequence%04d' % i, 'frames%02d.png' % t)
#             img = np.array(Image.open(img_path)) / 255.0  # normalize
#             seq_img[t] = img
#         np.save(os.path.join(dir, 'sequence%04d' % i, 'seq.npy'), seq_img)

# dir = 'Data/test_sequence/'
# for i in range(500):
#     if not os.path.exists(os.path.join(dir, 'sequence%03d' % i, 'seq.npy')):
#         seq_img = np.zeros([10, 64, 64, 3])
#         for t in range(10):
#             img_path = os.path.join(dir, 'sequence%03d' % i, 'frames%02d.png' % t)
#             img = np.array(Image.open(img_path)) / 255.0  # normalize
#             seq_img[t] = img
#         np.save(os.path.join(dir, 'sequence%03d' % i, 'seq.npy'), seq_img)

def get_train_batch(batch_size, n_seq=10000, T=10, K=10, dir='Data/train_sequence/'):
    selected_idx = random.sample(range(n_seq), batch_size)
    input = np.zeros([batch_size, T, 64, 64, 3])
    output = np.zeros([batch_size, K, 64, 64, 3])
    for i, idx in enumerate(selected_idx):
        seq_img = np.load(os.path.join(dir, 'sequence%04d' % idx, 'seq.npy'))
        input[i] = seq_img[:10]
        output[i] = seq_img[10:20]

    return input, output


def get_val_batch(start, end, T=10, K=10, dir='Data/val_sequence/'):
    input = np.zeros([end - start, T, 64, 64, 3])
    output = np.zeros([end - start, K, 64, 64, 3])
    for i, idx in enumerate(range(start, end)):
        seq_img = np.load(os.path.join(dir, 'sequence%03d' % idx, 'seq.npy'))
        input[i] = seq_img[:10]
        output[i] = seq_img[10:20]

    return input, output


def get_test_batch(start, end, T=10, dir='Data/test_sequence/'):
    input = np.zeros([end - start, T, 64, 64, 3])
    for i, idx in enumerate(range(start, end)):
        seq_img = np.load(os.path.join(dir, 'sequence%03d' % idx, 'seq.npy'))
        input[i] = seq_img

    return input