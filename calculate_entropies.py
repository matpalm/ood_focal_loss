import argparse

import h5py
import jax.numpy as jnp
import objax
import pandas as pd
import seaborn as sns
from objax.functional import softmax
from objax.zoo.resnet_v2 import ResNet18

import layers
import util as u

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-dir', type=str, required=True)
parser.add_argument('--include-temp-layer', action='store_true')  # o_O clumsy
opts = parser.parse_args()

model = ResNet18(in_channels=3, num_classes=10)
if opts.include_temp_layer:
    model.append(layers.Temperature())

objax.io.load_var_collection(f"{opts.model_dir}/weights.npz", model.vars())

predict = objax.Jit(lambda x: softmax(model(x, training=False)),
                    model.vars())

with h5py.File('splits.h5', 'r') as hf:
    x_train, _y_train = u.unpack_x_y(hf, 'train')
    x_validate, _y_validate = u.unpack_x_y(hf, 'validate')
    x_test, _y_test = u.unpack_x_y(hf, 'test')
    x_ood, _y_ood = u.unpack_x_y(hf, 'ood')

n = len(x_test)  # assume validate the same size too
y_pred_train = u.predict_in_batches(predict, x_train[:n], batch_size=64)
y_pred_validate = u.predict_in_batches(predict, x_validate, batch_size=64)
y_pred_test = u.predict_in_batches(predict, x_test, batch_size=64)
y_pred_ood = u.predict_in_batches(predict, x_ood[:n], batch_size=64)

df = pd.DataFrame({"train": u.entropy(y_pred_train),
                   "validate": u.entropy(y_pred_validate),
                   "test": u.entropy(y_pred_test),
                   "ood": u.entropy(y_pred_ood)})

df.to_csv(f"{opts.model_dir}/entropy.tsv", sep="\t", index=False)

p = sns.displot(df, kind='kde')
p.set(xlim=(0, None))
p.fig.set_figwidth(16)
p.fig.set_figwidth(9)
p.savefig(f"{opts.model_dir}/entropy.png", transparent=True)
