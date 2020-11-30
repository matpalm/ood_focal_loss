import argparse

import h5py
import numpy as np
import objax
from objax.functional import softmax
from objax.functional.loss import cross_entropy_logits_sparse
from objax.zoo.resnet_v2 import ResNet18

import util as u

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output-model', type=str, required=True)
opts = parser.parse_args()

model = ResNet18(in_channels=3, num_classes=10)

predict = objax.Jit(lambda x: softmax(model(x, training=False)),
                    model.vars())

optimiser = objax.optimizer.Adam(model.vars())


def loss_fn(x, y_true):
    logits = model(x, training=True)
    return cross_entropy_logits_sparse(logits, y_true).mean()


grad_values = objax.GradValues(loss_fn, model.vars())


def train_op(x, y_true, learning_rate):
    gradients, values = grad_values(x, y_true)
    optimiser(lr=learning_rate, grads=gradients)
    return values


train_op = objax.Jit(train_op, optimiser.vars() + grad_values.vars())


with h5py.File('splits.h5', 'r') as hf:
    x_train, y_train = u.unpack_x_y(hf, 'train')
    x_validate, y_validate = u.unpack_x_y(hf, 'validate')

batch_size = 64
learning_rate = 1e-3
best_validation_accuracy = 0

for epoch in range(10):

    # train one
    train_idxs = np.arange(len(x_train))
    np.random.shuffle(train_idxs)
    for batch_idx_offset in range(0, len(x_train), batch_size):
        batch_idxs = train_idxs[batch_idx_offset: batch_idx_offset+batch_size]
        loss_value = train_op(
            x_train[batch_idxs], y_train[batch_idxs], learning_rate)

    # report validation top1 accuracy
    accuracy = u.accuracy(predict, x_validate, y_validate)
    print("learning_rate", learning_rate,
          "validation accuracy", accuracy)

    # very simple step down across learning rates
    if epoch < 3:
        continue
    elif accuracy > best_validation_accuracy:
        best_validation_accuracy = accuracy
    elif learning_rate == learning_rate:
        learning_rate = 1e-4
    else:
        break

objax.io.save_var_collection(opts.output_model, model.vars())
