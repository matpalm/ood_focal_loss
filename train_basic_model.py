import argparse

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import objax
from objax.functional import softmax
from objax.functional.loss import cross_entropy_logits_sparse
from objax.zoo.resnet_v2 import ResNet18

import util as u

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--loss-fn', type=str, required=True)
parser.add_argument('--gamma', type=float, default=1.0,
                    help='gamma value for focal_loss')
parser.add_argument('--model-dir', type=str, required=True)
opts = parser.parse_args()

if opts.loss_fn not in ['cross_entropy', 'focal_loss']:
    raise Exception()

model = ResNet18(in_channels=3, num_classes=10)

predict = objax.Jit(lambda x: softmax(model(x, training=False)),
                    model.vars())

optimiser = objax.optimizer.Adam(model.vars())


def focal_loss_sparse(logits, y_true, gamma=1.0):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_probs = log_probs[jnp.arange(len(log_probs)), y_true]
    probs = jnp.exp(log_probs)
    elementwise_loss = -1 * ((1 - probs)**gamma) * log_probs
    return elementwise_loss


def loss_fn(x, y_true):
    logits = model(x, training=True)

    if opts.loss_fn == 'cross_entropy':
        return jnp.mean(cross_entropy_logits_sparse(logits, y_true))
    elif opts.loss_fn == 'focal_loss':
        return jnp.mean(focal_loss_sparse(logits, y_true, opts.gamma))
    else:
        raise Exception()


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

for epoch in range(20):

    # train one epoch
    train_idxs = np.arange(len(x_train))
    np.random.shuffle(train_idxs)
    for batch_idx_offset in range(0, len(x_train), batch_size):
        batch_idxs = train_idxs[batch_idx_offset: batch_idx_offset+batch_size]
        loss_value = train_op(
            x_train[batch_idxs], y_train[batch_idxs], learning_rate)

    # report validation top1 accuracy along with entropy
    accuracy, entropy = u.accuracy_and_entropy(predict, x_validate, y_validate)
    print(f"learning_rate {learning_rate} validation accuracy {accuracy:0.3f}",
          f"entropy min/mean/max {np.min(entropy):0.4f} {np.mean(entropy):0.4f}",
          f"{np.max(entropy):0.4f}")

    # simple warm up for a few epochs then continue while improving on
    # validation with one learning rate step down before early stopping
    if epoch < 3:
        continue
    elif accuracy > best_validation_accuracy:
        best_validation_accuracy = accuracy
    elif learning_rate == 1e-3:
        learning_rate = 1e-4
    else:
        break

model_weights = f"{opts.model_dir}/weights.npz"
u.ensure_dir_exists_for_file(model_weights)
objax.io.save_var_collection(model_weights, model.vars())
