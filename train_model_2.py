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
parser.add_argument('--input-model-dir', type=str, required=True)
parser.add_argument('--output-model-dir', type=str, required=True)
opts = parser.parse_args()

# load pre trained model

model = ResNet18(in_channels=3, num_classes=10)
input_model_weights = f"{opts.input_model_dir}/weights.npz"
objax.io.load_var_collection(input_model_weights, model.vars())

predict = objax.Jit(lambda x: softmax(model(x, training=False)),
                    model.vars())

# fine tune just last layer

classifier_layer = model[-1]

optimiser = objax.optimizer.Adam(classifier_layer.vars())


def loss_fn(x, y_true):
    # note we only want to optimise the last layer and not anything
    # about the rest of the resnet (including things like batchnorm)
    # see https://github.com/google/objax/issues/164
    logits = model(x, training=False)
    return cross_entropy_logits_sparse(logits, y_true).mean()


grad_values = objax.GradValues(loss_fn, classifier_layer.vars())


def train_op(x, y_true, learning_rate):
    gradients, values = grad_values(x, y_true)
    optimiser(lr=learning_rate, grads=gradients)
    return values


train_op = objax.Jit(train_op, optimiser.vars() + grad_values.vars())

with h5py.File('splits.h5', 'r') as hf:
    x_calibrate, y_calibrate = u.unpack_x_y(hf, 'calibrate')

batch_size = 64

for epoch in range(20):

    if epoch < 10:
        learning_rate = 1e-3
    else:
        learning_rate = 1e-4

    # train one epoch
    train_idxs = np.arange(len(x_calibrate))
    np.random.shuffle(train_idxs)
    for batch_idx_offset in range(0, len(x_calibrate), batch_size):
        batch_idxs = train_idxs[batch_idx_offset: batch_idx_offset+batch_size]
        loss_value = train_op(
            x_calibrate[batch_idxs], y_calibrate[batch_idxs], learning_rate)

    # report calibration top1 accuracy
    accuracy, entropy = u.accuracy_and_entropy(
        predict, x_calibrate, y_calibrate)
    print(f"learning_rate {learning_rate} calibration accuracy {accuracy:0.3f}",
          f"entropy min/mean/max {np.min(entropy):0.4f} {np.mean(entropy):0.4f}",
          f"{np.max(entropy):0.4f}")

output_model_weights = f"{opts.output_model_dir}/weights.npz"
u.ensure_dir_exists_for_file(output_model_weights)
objax.io.save_var_collection(output_model_weights, model.vars())
