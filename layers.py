import jax.numpy as jnp
import objax


class Temperature(objax.module.Module):

    def __init__(self):
        super().__init__()
        self.temperature = objax.variable.TrainVar(jnp.array([1.0]))

    def __call__(self, x):
        return x / self.temperature.value
