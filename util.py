import jax
import jax.numpy as jnp
import numpy as np
import os


def unpack_x_y(hf, split):
    x = hf["%s/x" % split][()]
    y = hf["%s/y" % split][()]
    return x, y


def predict_in_batches(predict_fn, x, batch_size=64):
    y_preds = []
    for offset in range(0, len(x), batch_size):
        batch = x[offset:offset+batch_size]
        y_preds.append(predict_fn(batch))
    return np.concatenate(y_preds)


def entropy(y):
    # per row entropy; i.e. (N, C) -> (N, )
    return jnp.sum(jax.scipy.special.entr(y), axis=1)


def accuracy_and_entropy(predict_fn, x, y_true, batch_size=64):
    y_probs = predict_in_batches(predict_fn, x, batch_size)
    y_preds = np.argmax(y_probs, axis=1)
    num_correct = np.sum(y_preds == y_true)
    accuracy = num_correct / len(y_true)
    return accuracy, entropy(y_probs)


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dir_exists_for_file(f):
    ensure_dir_exists(os.path.dirname(f))


def deciles(a):
    return np.percentile(a, np.linspace(0, 100, 11))
