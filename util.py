import jax
import jax.numpy as jnp
import numpy as np

labels = ["Airplane", "Automobile", "Bird",
          "Cat", "Deer", "Dog", "Frog", "Horse",
          "Ship", "Truck"]

label_to_idx = {l: i for i, l in enumerate(labels)}


def unpack_x_y(hf, split):
    x = hf["%s/x" % split][()]
    y = hf["%s/y" % split][()]
    return x, y


# TODO: use
# def shuffled_batch_idxs(n, batch_size=64):
#     idxs = np.arange(n)
#     np.random.shuffle(idxs)
#     for batch_idx_offset in range(0, n, batch_size):
#         yield idxs[batch_idx_offset: batch_idx_offset+batch_size]


def predict_in_batches(predict_fn, x, batch_size=64):
    y_preds = []
    for offset in range(0, len(x), batch_size):
        batch = x[offset:offset+batch_size]
        y_preds.append(predict_fn(batch))
    return np.concatenate(y_preds)


def entropy(y):
    # per row entropy; i.e. (N, C) -> (N, )
    return jnp.sum(jax.scipy.special.entr(y), axis=1)


def accuracy(predict_fn, x, y_true, batch_size=64):
    y_probs = predict_in_batches(predict_fn, x, batch_size)
    y_preds = np.argmax(y_probs, axis=1)
    num_correct = np.sum(y_preds == y_true)
    accuracy = num_correct / len(y_true)
    return accuracy
