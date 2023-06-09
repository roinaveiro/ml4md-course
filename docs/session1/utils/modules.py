from typing import List

import numpy as np
import sonnet as snt
import tensorflow as tf


def print_model(model):
    print(f'{model.__class__.__name__} : {model.name}\n')
    print(snt.format_variables(model.variables))
    n_params = np.sum([np.prod(v.shape) for v in model.variables])
    trainable_params = np.sum(
        [np.prod(v.shape) for v in model.trainable_variables])
    print(f'\nParams: {trainable_params} trainable out of {n_params}')


def get_linear_variables(model) -> List[tf.Variable]:
    """Gets all linear weight variables, useful for regularization."""
    weight_vars = []
    for v in model.trainable_variables:
        layer_name, _ = v.name.split('/')[-2:]
        if layer_name.startswith('linear_'):
            weight_vars.append(v)
    return weight_vars


def cast_activation(act):
    """Map string to activation, or just pass the activation function."""
    activations = {
        'relu': tf.nn.relu,
        'tanh': tf.nn.tanh,
        'sigmoid': tf.nn.sigmoid,
        'softmax': tf.nn.softmax,
        'identity': tf.identity
    }
    if callable(act):
        return act
    else:
        return activations[act]


def get_pred_layer(output_dim: int, output_act):
    return snt.Sequential([snt.Linear(output_dim), cast_activation(output_act)])


def get_mlp_fn(layer_sizes, act = 'relu'):
    """Instantiates a new MLP, followed by LayerNorm."""

    def make_mlp():
        return snt.Sequential([
            snt.nets.MLP(
                layer_sizes, activate_final=True, activation=cast_activation(act)),
            snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
        ])

    return make_mlp
