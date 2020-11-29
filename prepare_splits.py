from tensorflow.keras.datasets import cifar10
import numpy as np
import util as u
import h5py


# create h5 bundle
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
hf.create_dataset('x_ood', data=x[ood_idxs])
hf.create_dataset('y_ood', data=y[ood_idxs])

not_ood_idxs = np.logical_not(ood_idxs)
x_other = x[not_ood_idxs]
y_other = y[not_ood_idxs]


def slice_out(a, b):
    a, b = int(a), int(b)
    return x_other[a:b], y_other[a:b]


# make new train, validate, calibrate, test sets
n = len(x_other)
x_train, y_train = slice_out(0, 0.7*n)
x_validate, y_validate = slice_out(0.7*n, 0.8*n)
x_calibrate, y_calibrate = slice_out(0.8*n, 0.9*n)
x_test, y_test = slice_out(0.9*n, n)

hf.create_dataset('x_train', data=x_train)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('x_validate', data=x_validate)
hf.create_dataset('y_validate', data=y_validate)
hf.create_dataset('x_calibrate', data=x_calibrate)
hf.create_dataset('y_calibrate', data=y_calibrate)
hf.create_dataset('x_test', data=x_test)
hf.create_dataset('y_test', data=y_test)
hf.close()
