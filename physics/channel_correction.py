import numpy as np


def get_channel_correction(config_or_tx):
    correction = config_or_tx.get('channel_correction', {}) if config_or_tx else {}
    gains = np.asarray(correction.get('gains', np.ones(9)), dtype=float)
    biases = np.asarray(correction.get('biases_v', np.zeros(9)), dtype=float)
    if gains.shape != (9,) or biases.shape != (9,):
        raise ValueError('channel_correction gains and biases_v must each have 9 values')
    return gains, biases


def apply_channel_correction(emf_v, config_or_tx):
    gains, biases = get_channel_correction(config_or_tx)
    return np.asarray(emf_v, dtype=float) * gains + biases
