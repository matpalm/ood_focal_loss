import h5py
import numpy as np
from tensorflow.keras.datasets import cifar10

import util as u

# open h5 bundle
hf = h5py.File('splits.h5', 'w')

# fetch cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# combine provided train and test
x = np.concatenate([x_train, x_test])
y = np.squeeze(np.concatenate([y_train, y_test]))

# objax wants NCHW not NHWC
x = np.transpose(x, (0, 3, 1, 2))

# change dtype of input and normalise
x = x.astype(np.float32) / 255.0

# split out car & cat as ood examples vs everything else
ood_idxs = (y == u.label_to_idx['Automobile']) | (y == u.label_to_idx['Cat'])
hf.create_dataset('ood/x', data=x[ood_idxs])
hf.create_dataset('ood/y', data=y[ood_idxs])

not_ood_idxs = np.logical_not(ood_idxs)
x_other = x[not_ood_idxs]
y_other = y[not_ood_idxs]


def add_slice(key, a, b):
    a, b = int(a), int(b)
    hf.create_dataset("%s/x" % key, data=x_other[a:b])
    hf.create_dataset("%s/y" % key, data=y_other[a:b])


# make new train, validate, calibrate, test sets
n = len(x_other)
add_slice('train', 0, 0.7*n)
add_slice('validate', 0.7*n, 0.8*n)
add_slice('calibrate', 0.8*n, 0.9*n)
add_slice('test', 0.9*n, n)

# finish up
hf.close()
