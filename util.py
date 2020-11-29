import numpy as np

labels = ["Airplane", "Automobile", "Bird",
          "Cat", "Deer", "Dog", "Frog", "Horse",
          "Ship", "Truck"]

label_to_idx = {l: i for i, l in enumerate(labels)}


def unpack_x_y(hf, split):
    x = hf["%s/x" % split][()]
    y = hf["%s/y" % split][()]
    return x, y


def predict_in_batches(predict_fn, x, batch_size):
    y_preds = []
    for offset in range(0, len(x), batch_size):
        batch = x[offset:offset+batch_size]
        y_preds.append(predict_fn(batch))
    return np.concatenate(y_preds)
