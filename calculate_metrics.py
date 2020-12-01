import argparse

import h5py
import objax
import pandas as pd
import seaborn as sns
from objax.functional import softmax
from objax.zoo.resnet_v2 import ResNet18

import layers
import util as u

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--include-temp-layer', action='store_true')  # o_O clumsy
opts = parser.parse_args()

model = ResNet18(in_channels=3, num_classes=10)
if opts.include_temp_layer:
    model.append(layers.Temperature())

objax.io.load_var_collection(opts.model, model.vars())

predict = objax.Jit(lambda x: softmax(model(x, training=False)),
                    model.vars())

with h5py.File('splits.h5', 'r') as hf:
    x_train, y_train = u.unpack_x_y(hf, 'train')
    x_validate, y_validate = u.unpack_x_y(hf, 'validate')
    x_calibrate, y_calibrate = u.unpack_x_y(hf, 'calibrate')
    x_test, y_test = u.unpack_x_y(hf, 'test')
    x_ood, y_ood = u.unpack_x_y(hf, 'ood')


def calc_accuracy(x, y):
    accuracy, _entropy = u.accuracy_and_entropy(predict, x, y)
    return accuracy


n = len(x_test)  # assume validate & calibrate the same size too
print("train accuracy %0.3f" % calc_accuracy(x_train[:n], y_train[:n]))
print("validate accuracy %0.3f" % calc_accuracy(x_validate, y_validate))
print("calibrate accuracy %0.3f" % calc_accuracy(x_calibrate, y_calibrate))
print("test accuracy %0.3f" % calc_accuracy(x_test, y_test))
